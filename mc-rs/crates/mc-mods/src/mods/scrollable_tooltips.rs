//! Scrollable Tooltips: Scroll item tooltips that are too long for the screen.
//!
//! Keep the `id` stable (it is the settings-file key); the
//! `category` line must name the mod's home in `crate::menu_layout`, which decides where the
//! menu lists it (a test keeps the two in step).
//!
//! In an inventory or container, the mouse wheel moves an item tooltip that does not fit on the
//! screen (Shift + wheel moves it sideways); it snaps back when another item is hovered. The
//! wheel is only taken while the hovered tooltip overflows, so the creative inventory still
//! scrolls its items otherwise. Client-side only: nothing is sent.

use crate::category::Category;
use crate::descriptor::{ModDescriptor, OptionDescriptor, OptionKind, OptionValue};

static OPTIONS: [OptionDescriptor; 1] = [OptionDescriptor {
    id: "step",
    label: "Scroll step",
    kind: OptionKind::Slider {
        min: 4.0,
        max: 40.0,
        step: 2.0,
        unit: " px",
        decimals: 0,
    },
    default: OptionValue::Float(10.0),
    description: "How far one wheel notch moves the tooltip (10 px is one line).",
}];

/// The catalog entry.
pub static DESCRIPTOR: ModDescriptor = ModDescriptor {
    id: "scrollable_tooltips",
    name: "Scrollable Tooltips",
    category: Category::Interface,
    description: "Scroll item tooltips that are too long for the screen (Shift: sideways).",
    default_enabled: false,
    options: &OPTIONS,
    hud_elements: &[],
};
