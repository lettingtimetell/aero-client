//! Nametags: text shadow, your own nametag, background opacity and scale.
//!
//! Keep the `id` stable (it is the settings-file key);
//! the `category` line must name the mod's home in `crate::menu_layout`, which decides where the
//! menu lists it (a test keeps the two in step).
//!
//! Off is vanilla `Render.renderLivingLabel`: no shadow, a `color(0, 0, 0, 0.25)` backdrop, the
//! `0.016666668 * 1.6` label scale, and never your own name (`canRenderName` excludes
//! `renderManager.livingPlayer`). On, the options apply to every entity label (players, named
//! mobs, holograms); sign text is not a nametag and is untouched. The style itself lives in
//! `mc_render::entity::nameplate::NameplateStyle`; the client half is
//! `crates/mc-client/src/app/mods/x6_entities.rs`.
//!
//! Nametags can be made smaller and lighter than vanilla, never
//! larger or darker. Scale tops out at 1.00x ([`SCALE_MAX`]) and Background darkness at 25%
//! ([`BACKGROUND_MAX`]); older files with bigger values are clamped by the descriptor on load.
//!
//! **Always show mob names** ([`ALWAYS_MOB_NAMES`], off by default).
//! Vanilla 1.8.9 shows a mob's custom name only under the crosshair: `RenderLiving.canRenderName`
//! asks `getAlwaysRenderNameTagForRender() || hasCustomName() && entity == pointedEntity`, and
//! `EntityLivingBase.getAlwaysRenderNameTagForRender` is hard-coded `false` (so in 1.8.9 even the
//! server's CustomNameVisible flag cannot show it). On, every named mob is labelled the way a
//! player is: the same 64-block range (32 while sneaking), and the same `canRenderName` gate (F1
//! hides it; an invisible or ridden mob gets none). Players already show their names and their
//! team rules stay the server's; armor stands keep their own CustomNameVisible rule, so server
//! holograms neither change nor double. The client half is `app/mods/w18_fb.rs`.

use crate::category::Category;
use crate::descriptor::{ModDescriptor, OptionDescriptor, OptionKind, OptionValue};

/// "Always show mob names"'s option id. A settings-file key: never
/// rename it.
pub const ALWAYS_MOB_NAMES: &str = "always_mob_names";

static OPTIONS: [OptionDescriptor; 5] = [
    OptionDescriptor {
        id: "text_shadow",
        label: "Text shadow",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(true),
        description: "Draw names with the font's drop shadow, like chat.",
    },
    OptionDescriptor {
        id: "own_nametag",
        label: "Show own nametag",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(true),
        description: "Show your own name above your head in third person (F5).",
    },
    // Beside "Show own nametag": both choose whose names show. The
    // order is only the menu's; the settings file keys options by id.
    OptionDescriptor {
        id: ALWAYS_MOB_NAMES,
        label: "Always show mob names",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(false),
        description: "Show every named mob's nametag like a player's, without aiming at it. \
                      Players and holograms are unchanged.",
    },
    OptionDescriptor {
        id: "background",
        label: "Background darkness",
        kind: OptionKind::Slider {
            min: 0.0,
            max: BACKGROUND_MAX,
            step: 1.0,
            unit: "%",
            decimals: 0,
        },
        default: OptionValue::Float(25.0),
        description: "Black backdrop opacity, up to vanilla's 25%. 0% removes it.",
    },
    OptionDescriptor {
        id: "scale",
        label: "Scale",
        kind: OptionKind::Slider {
            min: 0.5,
            max: SCALE_MAX,
            step: 0.05,
            unit: "x",
            decimals: 2,
        },
        default: OptionValue::Float(1.0),
        description: "Size of every nametag, down to 0.50x. 1.00x is vanilla and the largest.",
    },
];

/// "Scale"'s maximum: vanilla's size (by design the mod never makes nametags larger than
/// vanilla). A saved value above it is clamped on load.
pub const SCALE_MAX: f32 = 1.0;

/// "Background darkness"'s maximum: vanilla's `color(0, 0, 0, 0.25)` (never darker than vanilla).
/// A saved value above it is clamped on load.
pub const BACKGROUND_MAX: f32 = 25.0;

/// The catalog entry.
pub static DESCRIPTOR: ModDescriptor = ModDescriptor {
    id: "nametags",
    name: "Nametags",
    category: Category::Players,
    description: "Text shadow, your own nametag in third person, named mobs, background opacity \
                  and size.",
    default_enabled: false,
    options: &OPTIONS,
    hud_elements: &[],
};
