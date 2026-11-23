//! Menu Blur: blurs the game behind the pause menu, the options and inventories.
//!
//! Keep the `id` stable (it is the settings-file key); the
//! `category` line must name the mod's home in `crate::menu_layout`, which decides where the
//! menu lists it (a test keeps the two in step).
//!
//! The effect is `mc_render::post::blur::MenuBlurPhase` (a dual-Kawase blur at
//! `PhaseOrder::MENU_BLUR`, between the HUD and the open screen); the client glue is
//! `mc-client/src/app/mods/x10_blur.rs`. Chat, the death screen and the Aero mod menu (whose glass
//! frosts the game itself) are never blurred. Vanilla's dark menu gradient still draws on top.

use crate::category::Category;
use crate::descriptor::{ModDescriptor, OptionDescriptor, OptionKind, OptionValue};

static OPTIONS: [OptionDescriptor; 4] = [
    OptionDescriptor {
        id: "radius",
        label: "Radius",
        kind: OptionKind::Slider {
            min: 2.0,
            max: 48.0,
            step: 1.0,
            unit: " px",
            decimals: 0,
        },
        default: OptionValue::Float(20.0),
        description: "How far the blur reaches, in pixels at 1080p (it scales with the window).",
    },
    OptionDescriptor {
        id: "fade_in",
        label: "Fade-in",
        kind: OptionKind::Slider {
            min: 0.0,
            max: 500.0,
            step: 10.0,
            unit: " ms",
            decimals: 0,
        },
        default: OptionValue::Float(150.0),
        description: "How long the blur takes to build up when a menu opens. 0 is instant.",
    },
    OptionDescriptor {
        id: "blur_hud",
        label: "Blur HUD",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(true),
        description: "Also blur the hotbar, chat and scoreboard behind the menu. \
                      Off blurs only the world.",
    },
    OptionDescriptor {
        id: "inventories",
        label: "In inventories",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(true),
        description: "Blur behind the inventory, chests and other containers too. Off keeps the \
                      world sharp while you loot.",
    },
];

/// The catalog entry.
pub static DESCRIPTOR: ModDescriptor = ModDescriptor {
    id: "menu_blur",
    name: "Menu Blur",
    category: Category::Graphics,
    description: "Blurs the game behind the pause menu, the options and inventories (never \
                  chat). Costs nothing while no menu is open.",
    default_enabled: false,
    options: &OPTIONS,
    hud_elements: &[],
};
