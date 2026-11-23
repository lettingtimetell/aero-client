//! Super Secret Settings: vanilla's hidden post-processing shaders, picked from a list.
//!
//! Keep the `id` stable (it is the settings-file key); the
//! `category` line must name the mod's home in `crate::menu_layout`, which decides where the
//! menu lists it (a test keeps the two in step).
//!
//! Vanilla's "Super Secret Settings..." button (`GuiOptions`, id 8675309) stepped through
//! `EntityRenderer.shaderResourceLocations` one random-looking press at a time; this mod replaces
//! it (the button is gone from the options screen). Choice index `i > 0` is
//! `mc_render::post::SUPER_SECRET_SHADER_GROUPS[i - 1]`: the labels below are those 24 group
//! stems in vanilla's order, made readable. The choice is persisted **by label**, so never
//! rename one. Client glue: `mc-client/src/app/mods/x10_blur.rs`.
//!
//! Aero's own effects follow vanilla's groups, from
//! [`FIRST_AERO_CHOICE`] on. "ASCII" redraws the world as Minecraft-font characters
//! (`mc_render::post::ascii`, client glue `mc-client/src/app/mods/nw_rfx.rs`), styled by
//! "Character size" and "Character colors", which the menu shows only while ASCII is chosen.

use crate::category::Category;
use crate::descriptor::{ModDescriptor, OptionDescriptor, OptionKind, OptionValue};

/// The first choice index that is not one of vanilla's groups: `SHADER_CHOICES[1..25]` are
/// vanilla's 24 groups.
pub const FIRST_AERO_CHOICE: usize = 25;

/// The "ASCII" choice's index.
pub const ASCII_CHOICE: usize = 25;

/// "Character size" choices (`mc_render::post::ascii::AsciiSize`).
pub const ASCII_SIZES: [&str; 3] = ["Small", "Medium", "Large"];

/// "Character colors" choices (`mc_render::post::ascii::AsciiColors`).
pub const ASCII_COLORS: [&str; 3] = ["Color", "White", "Green"];

/// "Off", then vanilla's 24 shader groups in `EntityRenderer.shaderResourceLocations` order, then
/// Aero's own effects (from [`FIRST_AERO_CHOICE`]).
pub const SHADER_CHOICES: [&str; 26] = [
    "Off",
    "Notch",
    "FXAA",
    "Art",
    "Bumpy",
    "Blobs 2",
    "Pencil",
    "Color Convolve",
    "Deconverge",
    "Flip",
    "Invert",
    "NTSC",
    "Outline",
    "Phosphor",
    "Scan Pincushion",
    "Sobel",
    "Bits",
    "Desaturate",
    "Green",
    "Blur",
    "Wobble",
    "Blobs",
    "Antialias",
    "Creeper",
    "Spider",
    // Aero's own effects.
    "ASCII",
];

static OPTIONS: [OptionDescriptor; 3] = [
    OptionDescriptor {
        id: "shader",
        label: "Shader",
        kind: OptionKind::Choice {
            choices: &SHADER_CHOICES,
        },
        default: OptionValue::Choice(0),
        description: "Click to step forward, right-click to step back. Applies at once and again \
                      every time you join a world. F4 does not hide it. ASCII (the last one) \
                      redraws the world as Minecraft-font characters.",
    },
    OptionDescriptor {
        id: "ascii_size",
        label: "Character size",
        kind: OptionKind::Choice {
            choices: &ASCII_SIZES,
        },
        default: OptionValue::Choice(1),
        description: "How big the ASCII characters are. The sizes keep a similar number of rows \
                      at any window size.",
    },
    OptionDescriptor {
        id: "ascii_colors",
        label: "Character colors",
        kind: OptionKind::Choice {
            choices: &ASCII_COLORS,
        },
        default: OptionValue::Choice(0),
        description: "Color draws each character in the color of the world behind it; White \
                      and Green are classic terminal looks on black.",
    },
];

/// The catalog entry.
pub static DESCRIPTOR: ModDescriptor = ModDescriptor {
    id: "super_secret_settings",
    name: "Super Secret Settings",
    category: Category::Graphics,
    description: "Vanilla's hidden post-processing shaders (Notch, Art, Pencil, Creeper...), \
                  picked from a list instead of the random options button, plus ASCII, which \
                  redraws the world as characters. The HUD stays sharp.",
    default_enabled: false,
    options: &OPTIONS,
    hud_elements: &[],
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ascii_follows_vanillas_groups_and_its_styles_default_to_medium_color() {
        assert_eq!(SHADER_CHOICES[0], "Off");
        assert_eq!(SHADER_CHOICES[FIRST_AERO_CHOICE - 1], "Spider");
        assert_eq!(SHADER_CHOICES[ASCII_CHOICE], "ASCII");
        assert_eq!(SHADER_CHOICES.len(), ASCII_CHOICE + 1);
        let ids: Vec<_> = DESCRIPTOR.options.iter().map(|o| o.id).collect();
        assert_eq!(ids, ["shader", "ascii_size", "ascii_colors"]);
        assert_eq!(DESCRIPTOR.options[0].default, OptionValue::Choice(0));
        assert_eq!(ASCII_SIZES[1], "Medium");
        assert_eq!(DESCRIPTOR.options[1].default, OptionValue::Choice(1));
        assert_eq!(ASCII_COLORS[0], "Color");
        assert_eq!(DESCRIPTOR.options[2].default, OptionValue::Choice(0));
    }
}
