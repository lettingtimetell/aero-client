//! Hitboxes: show and style entity hitboxes.
//!
//! Keep
//! the `id` stable (it is the settings-file key); the `category` line must name the mod's home in
//! `crate::menu_layout`, which decides where the menu lists it (a test keeps the two in step).
//!
//! The mod is a restyled F3+B (`RenderManager.renderDebugBoundingBox`): the same box, the same
//! eye-height line and two-block look vector, depth-tested like vanilla (never through walls),
//! never for invisible entities, your own box only in third person. While it is on it replaces
//! vanilla's thin F3+B lines; "Show" picks whether its boxes are always there or follow F3+B.
//! Lines are screen-facing quads (wgpu has no wide lines), so the width is real pixels. The
//! renderer is `mc_render::entity::hitbox`; the client half is
//! `crates/mc-client/src/app/mods/x6_entities.rs`.

use crate::category::Category;
use crate::descriptor::{ModDescriptor, OptionDescriptor, OptionKind, OptionValue};

static OPTIONS: [OptionDescriptor; 10] = [
    OptionDescriptor {
        id: "show",
        label: "Show",
        kind: OptionKind::Choice {
            choices: &["Always", "With F3+B"],
        },
        default: OptionValue::Choice(0),
        description: "Always, or only while F3+B is toggled on.",
    },
    OptionDescriptor {
        id: "players",
        label: "Players",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(true),
        description: "Other players, and yourself in third person.",
    },
    OptionDescriptor {
        id: "mobs",
        label: "Mobs",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(false),
        description: "Animals, monsters, villagers, golems and armor stands.",
    },
    OptionDescriptor {
        id: "items",
        label: "Items",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(false),
        description: "Dropped items and experience orbs.",
    },
    OptionDescriptor {
        id: "projectiles",
        label: "Projectiles",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(false),
        description: "Arrows, snowballs, eggs, pearls, potions, fireballs and fishing hooks.",
    },
    OptionDescriptor {
        id: "other",
        label: "Other entities",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(false),
        description: "Minecarts, boats, TNT, falling blocks, paintings and item frames.",
    },
    OptionDescriptor {
        id: "color",
        label: "Box color",
        kind: OptionKind::Color { alpha: true },
        default: OptionValue::Color(0xFFFF_FFFF),
        description: "Vanilla F3+B is white.",
    },
    OptionDescriptor {
        id: "width",
        label: "Line width",
        kind: OptionKind::Slider {
            min: 1.0,
            max: 6.0,
            step: 0.5,
            unit: " px",
            decimals: 1,
        },
        default: OptionValue::Float(2.0),
        description: "Thickness of every line, in screen pixels.",
    },
    OptionDescriptor {
        id: "eye_line",
        label: "Eye height line",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(true),
        description: "The red line at a living entity's eye height.",
    },
    OptionDescriptor {
        id: "look_vector",
        label: "Look vector",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(true),
        description: "The blue line, two blocks long, showing where the entity looks.",
    },
];

/// The catalog entry.
pub static DESCRIPTOR: ModDescriptor = ModDescriptor {
    id: "hitboxes",
    name: "Hitboxes",
    category: Category::Pvp,
    description: "Shows entity hitboxes, like F3+B, in your own color and line width.",
    default_enabled: false,
    options: &OPTIONS,
    hud_elements: &[],
};
