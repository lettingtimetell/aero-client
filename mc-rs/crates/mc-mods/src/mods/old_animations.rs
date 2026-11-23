//! 1.7 Animations: the 1.7 held-item look and feel on a 1.8.9 client.
//!
//! Keep the `id` stable (it is the settings-file key); the
//! `category` line must name the mod's home in `crate::menu_layout`, which decides where the
//! menu lists it (a test keeps the two in step).
//!
//! Every option is visual only, none of them changes a packet, except one: `use_stops_mining`
//! (it moved here from Modern Keybinds' "Eating stops mining"
//! and grew to swords and bows). A use press during a dig then sends vanilla's own "let go of
//! attack" C07 `ABORT_DESTROY_BLOCK` and the right-click's C08s in the same tick, where vanilla
//! needs two ticks and a released attack button. Its client half is `x4_use_stops_mining` in
//! `mc-client/src/app/mods/x4_input.rs`. The render maths lives in
//! `mc-render/src/pipeline/old_animations.rs`; the client glue in
//! `mc-client/src/app/mods/m5_animations.rs`. The option list follows

use crate::category::Category;
use crate::descriptor::{ModDescriptor, OptionDescriptor, OptionKind, OptionValue};

/// The sub-options, in menu order. All on once the mod is enabled, except `old_health`, which
/// hides information (how much health a hit took) rather than restoring an animation.
///
/// A slice, so there is no length to keep in step.
static OPTIONS: &[OptionDescriptor] = &[
    OptionDescriptor {
        id: "blockhit",
        label: "Block-hit swing",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(true),
        description: "The sword keeps swinging while it blocks, as in 1.7.",
    },
    OptionDescriptor {
        id: "punch_during_use",
        label: "Swing at blocks while using",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(true),
        description: "Holding attack on a block while blocking or eating shows a swing. \
                      Visual only: no packet is sent.",
    },
    OptionDescriptor {
        id: "item_positions",
        label: "1.7 held item position",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(true),
        description: "Flat items (swords, tools, food) sit bigger, lower and further right.",
    },
    OptionDescriptor {
        id: "bow",
        label: "1.7 bow pose",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(true),
        description: "The drawn bow stretches along its own length, as in 1.7.",
    },
    OptionDescriptor {
        id: "rod",
        label: "1.7 fishing rod position",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(true),
        description: "Fishing rods and carrots on a stick are held the 1.7 way round.",
    },
    OptionDescriptor {
        id: "sword_block_3p",
        label: "1.7 third-person sword block",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(true),
        description: "Other players (and you in F5) block with the 1.7 arm and sword pose.",
    },
    OptionDescriptor {
        id: "armor_red",
        label: "Armor turns red when hit",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(true),
        description: "Hit players flash red over their armor too, which makes hits easier to see.",
    },
    OptionDescriptor {
        id: "smooth_sneak",
        label: "Smooth sneak camera",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(true),
        description: "The camera glides when you stand up from sneaking. Camera only.",
    },
    OptionDescriptor {
        id: "old_health",
        label: "No lost-health ghost hearts",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(false),
        description: "Hides the white hearts that show how much health a hit took.",
    },
    OptionDescriptor {
        id: "use_stops_mining",
        label: "Using items stops mining",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(true),
        description: "Blocking with a sword, drawing a bow, eating or drinking while you mine \
                      stops mining at once, as a right-click did in 1.7. Off: let go of attack \
                      first, as in 1.8.9. Changes packet timing.",
    },
    // MC-70336. The id stays `translucent_held` (ids are settings-file keys), but
    // the text says what 1.7 really did: it drew held items without blending, and its per-texel
    // side strips left no gaps, while 1.8.9 blends them and builds the sides from alpha == 0 only,
    // leaving see-through gaps beside faint (alpha 1..25) edge pixels. See
    // `mc-render/src/pipeline/old_item_edges.rs`.
    OptionDescriptor {
        id: "translucent_held",
        label: "1.7 item edges",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(true),
        description: "Held items with partly see-through pixels are drawn the 1.7 way: solid, \
                      with no see-through gaps along their edges (MC-70336), so older resource \
                      packs look right. Other items are unchanged.",
    },
];

/// The catalog entry.
pub static DESCRIPTOR: ModDescriptor = ModDescriptor {
    id: "old_animations",
    name: "1.7 Animations",
    category: Category::Pvp,
    description: "The 1.7 block-hit, held item positions, bow and rod poses, and red armor, \
                  all visual. One setting is not: Using items stops mining.",
    default_enabled: false,
    options: OPTIONS,
    hud_elements: &[],
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_option_is_a_toggle_and_only_old_health_starts_off() {
        assert_eq!(DESCRIPTOR.id, "old_animations");
        assert!(!DESCRIPTOR.default_enabled);
        for option in DESCRIPTOR.options {
            assert_eq!(option.kind, OptionKind::Toggle, "{}", option.id);
            let expected = option.id != "old_health";
            assert_eq!(option.default, OptionValue::Bool(expected), "{}", option.id);
        }
        let ids: Vec<_> = DESCRIPTOR.options.iter().map(|option| option.id).collect();
        assert_eq!(
            ids,
            [
                "blockhit",
                "punch_during_use",
                "item_positions",
                "bow",
                "rod",
                "sword_block_3p",
                "armor_red",
                "smooth_sneak",
                "old_health",
                "use_stops_mining",
                "translucent_held",
            ]
        );
    }
}
