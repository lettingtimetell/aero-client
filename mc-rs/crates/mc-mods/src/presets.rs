//! Option descriptors shared by the HUD mods, so every text readout has the same option ids and
//! `mc_ui::modhud::HudTextStyle::read` can build its style from any of them.
//!
//! Use them by value in a mod's option list:
//!
//! ```
//! use mc_mods::presets;
//! use mc_mods::OptionDescriptor;
//! static OPTIONS: [OptionDescriptor; 4] = [
//!     presets::TEXT_COLOR,
//!     presets::TEXT_SHADOW,
//!     presets::BACKGROUND,
//!     presets::BACKGROUND_COLOR,
//! ];
//! assert_eq!(OPTIONS[0].id, presets::TEXT_COLOR_ID);
//! ```

use crate::descriptor::{OptionDescriptor, OptionKind, OptionValue};

/// Id of [`TEXT_COLOR`].
pub const TEXT_COLOR_ID: &str = "text_color";
/// Id of [`TEXT_SHADOW`].
pub const TEXT_SHADOW_ID: &str = "text_shadow";
/// Id of [`BACKGROUND`].
pub const BACKGROUND_ID: &str = "background";
/// Id of [`BACKGROUND_COLOR`].
pub const BACKGROUND_COLOR_ID: &str = "background_color";
/// Id of [`BRACKETS`].
pub const BRACKETS_ID: &str = "brackets";

/// Text colour, white by default.
pub const TEXT_COLOR: OptionDescriptor = OptionDescriptor {
    id: TEXT_COLOR_ID,
    label: "Text color",
    kind: OptionKind::Color { alpha: true },
    default: OptionValue::Color(0xFFFF_FFFF),
    description: "",
};

/// Minecraft's one-pixel drop shadow under the text, on by default.
pub const TEXT_SHADOW: OptionDescriptor = OptionDescriptor {
    id: TEXT_SHADOW_ID,
    label: "Text shadow",
    kind: OptionKind::Toggle,
    default: OptionValue::Bool(true),
    description: "",
};

/// The translucent box behind the readout, on by default.
pub const BACKGROUND: OptionDescriptor = OptionDescriptor {
    id: BACKGROUND_ID,
    label: "Background",
    kind: OptionKind::Toggle,
    default: OptionValue::Bool(true),
    description: "",
};

/// Colour of that box: black at about 44 % by default.
pub const BACKGROUND_COLOR: OptionDescriptor = OptionDescriptor {
    id: BACKGROUND_COLOR_ID,
    label: "Background color",
    kind: OptionKind::Color { alpha: true },
    default: OptionValue::Color(0x6F00_0000),
    description: "",
};

/// Wrap the readout in `[ ]`, off by default.
pub const BRACKETS: OptionDescriptor = OptionDescriptor {
    id: BRACKETS_ID,
    label: "Brackets",
    kind: OptionKind::Toggle,
    default: OptionValue::Bool(false),
    description: "",
};
