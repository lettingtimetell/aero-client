//! Chat Macros: send a chat message or command with one key.
//!
//! Keep the `id`s stable (they are the settings-file keys).
//!
//! [`SLOTS`] slots, each a key plus a text. A press of a slot's key in game (no screen open) sends
//! that text exactly as typing it into chat and pressing Enter would: the chat input's character
//! filter, vanilla's trim, the sent-history entry, one `C01PacketChatMessage` cut at the chat
//! limit (256, always on). One send per press: holding the key (or the
//! OS key repeat) sends nothing more. An unbound key or an empty text leaves the slot idle.
//!
//! The client side is `crates/mc-client/src/app/mods/x2_chat.rs`.

use crate::category::Category;
use crate::descriptor::{ModDescriptor, OptionDescriptor, OptionKind, OptionValue};

/// How many key + text slots the mod has.
pub const SLOTS: usize = 6;

/// The longest text a slot holds, in characters. Sending cuts it to the chat limit.
pub const TEXT_MAX: usize = 256;

/// Option ids, slot by slot: `(key, text)`.
pub const SLOT_IDS: [(&str, &str); SLOTS] = [
    ("key1", "text1"),
    ("key2", "text2"),
    ("key3", "text3"),
    ("key4", "text4"),
    ("key5", "text5"),
    ("key6", "text6"),
];

const KEY_DESCRIPTION: &str = "Sends this macro's text. In game only, once per press.";
const TEXT_DESCRIPTION: &str = "A chat message or a /command.";

macro_rules! key {
    ($id:literal, $label:literal) => {
        OptionDescriptor {
            id: $id,
            label: $label,
            kind: OptionKind::Keybind,
            default: OptionValue::Key(0),
            description: KEY_DESCRIPTION,
        }
    };
}

macro_rules! text {
    ($id:literal, $label:literal) => {
        OptionDescriptor {
            id: $id,
            label: $label,
            kind: OptionKind::Text { max_len: TEXT_MAX },
            default: OptionValue::Text(std::borrow::Cow::Borrowed("")),
            description: TEXT_DESCRIPTION,
        }
    };
}

static OPTIONS: [OptionDescriptor; SLOTS * 2] = [
    key!("key1", "Macro 1 key"),
    text!("text1", "Macro 1 text"),
    key!("key2", "Macro 2 key"),
    text!("text2", "Macro 2 text"),
    key!("key3", "Macro 3 key"),
    text!("text3", "Macro 3 text"),
    key!("key4", "Macro 4 key"),
    text!("text4", "Macro 4 text"),
    key!("key5", "Macro 5 key"),
    text!("text5", "Macro 5 text"),
    key!("key6", "Macro 6 key"),
    text!("text6", "Macro 6 text"),
];

/// The catalog entry.
pub static DESCRIPTOR: ModDescriptor = ModDescriptor {
    id: "chat_macros",
    name: "Chat Macros",
    category: Category::Interface,
    description: "Send a chat message or /command with one key, in game, once per press.",
    default_enabled: false,
    options: &OPTIONS,
    hud_elements: &[],
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_slot_is_a_key_then_its_text() {
        assert_eq!(OPTIONS.len(), SLOTS * 2);
        for (slot, (key, text)) in SLOT_IDS.iter().enumerate() {
            let k = &OPTIONS[slot * 2];
            let t = &OPTIONS[slot * 2 + 1];
            assert_eq!((k.id, t.id), (*key, *text));
            assert!(matches!(k.kind, OptionKind::Keybind));
            assert_eq!(k.default, OptionValue::Key(0), "unbound by default");
            assert!(matches!(t.kind, OptionKind::Text { max_len: TEXT_MAX }));
            assert_eq!(t.default, OptionValue::Text("".into()));
            assert!(k.label.contains(&(slot + 1).to_string()));
        }
        assert!(!DESCRIPTOR.default_enabled);
    }
}
