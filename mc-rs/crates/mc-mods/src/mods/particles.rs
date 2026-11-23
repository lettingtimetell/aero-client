//! Particles: Scales the amount and size of specific particles: crits, potions, lava and more.
//!
//! Keep the `id` stable (it is the settings-file key); the
//! `category` line must name the mod's home in `crate::menu_layout`, which decides where the
//! menu lists it (a test keeps the two in step).
//!
//! # Menu layout
//!
//! Seven groups with an amount and a size each would be fourteen sliders, too many rows for one
//! menu window. The menu therefore shows a group picker (`group`) and one `amount` and one
//! `size` slider that edit the picked group. Every group's values persist in the hidden `table`
//! option, as `crit=200/100,sharp=200/100,...` (amount and size in whole percent). The client
//! (`app/mods/m6_particles.rs`) keeps the sliders and the table in step: picking a group loads
//! its row into the sliders, and moving a slider writes the row back.

use std::borrow::Cow;
use std::fmt::Write as _;

use crate::category::Category;
use crate::descriptor::{snap_slider, ModDescriptor, OptionDescriptor, OptionKind, OptionValue};

/// How many particle groups the table holds.
pub const GROUP_COUNT: usize = 7;

/// Each group's key in the `table` text, in `group` choice order. The client checks them
/// against `mc_render::particle::tuning::ParticleGroup::key`.
pub const GROUP_KEYS: [&str; GROUP_COUNT] = [
    "crit",
    "sharp",
    "swirl",
    "splash",
    "lava",
    "block",
    "explosion",
];

/// The `group` choice labels, in [`GROUP_KEYS`] order.
pub const GROUP_LABELS: [&str; GROUP_COUNT] = [
    "Crits",
    "Sharpness",
    "Potion swirls",
    "Potion splash",
    "Lava",
    "Block breaking",
    "Explosions",
];

/// Smallest `amount`, in percent (hides the group).
pub const AMOUNT_MIN: f32 = 0.0;
/// Largest `amount`, in percent (10x).
pub const AMOUNT_MAX: f32 = 1000.0;
/// `amount` slider step, in percent.
pub const AMOUNT_STEP: f32 = 25.0;
/// Smallest `size`, in percent.
pub const SIZE_MIN: f32 = 25.0;
/// Largest `size`, in percent.
pub const SIZE_MAX: f32 = 300.0;
/// `size` slider step, in percent.
pub const SIZE_STEP: f32 = 5.0;

/// One group's values in whole percent; `100/100` is vanilla.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GroupPercent {
    /// Particles per spawn attempt (`0` hides the group). For crits and sharpness, above
    /// `100` it is emitters per hit you land.
    pub amount: u16,
    /// Render size.
    pub size: u16,
}

impl GroupPercent {
    /// Vanilla: every particle once, at its own size.
    pub const VANILLA: GroupPercent = GroupPercent {
        amount: 100,
        size: 100,
    };

    /// Slider values (percent) clamped and snapped the way the two sliders store them.
    #[must_use]
    pub fn from_sliders(amount: f32, size: f32) -> GroupPercent {
        GroupPercent {
            amount: whole_percent(snap_slider(amount, AMOUNT_MIN, AMOUNT_MAX, AMOUNT_STEP)),
            size: whole_percent(snap_slider(size, SIZE_MIN, SIZE_MAX, SIZE_STEP)),
        }
    }
}

/// A snapped slider value as whole percent (a non-finite value reads as vanilla's 100).
fn whole_percent(value: f32) -> u16 {
    if value.is_finite() {
        value.round().clamp(0.0, f32::from(u16::MAX)) as u16
    } else {
        100
    }
}

/// The hit-particle groups' default: two emitters per hit you land.
const HIT_DEFAULT: GroupPercent = GroupPercent {
    amount: 200,
    size: 100,
};

/// Every group's default: crits and sharpness at 200%, everything else vanilla.
pub const DEFAULT_TABLE: [GroupPercent; GROUP_COUNT] = [
    HIT_DEFAULT,
    HIT_DEFAULT,
    GroupPercent::VANILLA,
    GroupPercent::VANILLA,
    GroupPercent::VANILLA,
    GroupPercent::VANILLA,
    GroupPercent::VANILLA,
];

/// [`DEFAULT_TABLE`] as the `table` option's text.
pub const DEFAULT_TABLE_TEXT: &str = "crit=200/100,sharp=200/100,swirl=100/100,splash=100/100,\
                                      lava=100/100,block=100/100,explosion=100/100";

/// Read the `table` text. Entries are `key=amount/size`, separated by commas. A group whose
/// entry is missing or malformed keeps its default, and values are clamped and snapped to the
/// sliders' ranges, so a hand-edited file cannot break anything.
#[must_use]
pub fn parse_table(text: &str) -> [GroupPercent; GROUP_COUNT] {
    let mut table = DEFAULT_TABLE;
    for entry in text.split(',') {
        let Some((key, values)) = entry.split_once('=') else {
            continue;
        };
        let Some(index) = GROUP_KEYS.iter().position(|k| *k == key.trim()) else {
            continue;
        };
        let Some((amount, size)) = values.split_once('/') else {
            continue;
        };
        let (Ok(amount), Ok(size)) = (amount.trim().parse::<f32>(), size.trim().parse::<f32>())
        else {
            continue;
        };
        if amount.is_finite() && size.is_finite() {
            table[index] = GroupPercent::from_sliders(amount, size);
        }
    }
    table
}

/// Write `table` as the `table` option's text into `out` (cleared first): the inverse of
/// [`parse_table`].
pub fn write_table(table: &[GroupPercent; GROUP_COUNT], out: &mut String) {
    out.clear();
    for (i, (key, group)) in GROUP_KEYS.iter().zip(table).enumerate() {
        if i > 0 {
            out.push(',');
        }
        let _ = write!(out, "{key}={}/{}", group.amount, group.size);
    }
}

static OPTIONS: [OptionDescriptor; 8] = [
    OptionDescriptor {
        id: "group",
        label: "Particle",
        kind: OptionKind::Choice {
            choices: &GROUP_LABELS,
        },
        default: OptionValue::Choice(0),
        description: "Picks the particles that Amount and Size edit.",
    },
    OptionDescriptor {
        id: "amount",
        label: "Amount",
        kind: OptionKind::Slider {
            min: AMOUNT_MIN,
            max: AMOUNT_MAX,
            step: AMOUNT_STEP,
            unit: "%",
            decimals: 0,
        },
        // Group 0 (crits) is shown first, and its default is 200%.
        default: OptionValue::Float(200.0),
        description:
            "How many of the picked particles appear: 100% is vanilla, 0% hides them. For \
                      crits and sharpness, emitters per hit you land. Particles: Minimal still \
                      applies.",
    },
    OptionDescriptor {
        id: "size",
        label: "Size",
        kind: OptionKind::Slider {
            min: SIZE_MIN,
            max: SIZE_MAX,
            step: SIZE_STEP,
            unit: "%",
            decimals: 0,
        },
        default: OptionValue::Float(100.0),
        description: "How big the picked particles draw. 100% is vanilla.",
    },
    OptionDescriptor {
        id: "always_crit",
        label: "Always show crits",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(false),
        description: "Crit stars on every hit you land, not only on critical hits.",
    },
    OptionDescriptor {
        id: "always_sharp",
        label: "Always show sharpness",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(true),
        description: "Sharpness sparkles on every hit you land, with any weapon.",
    },
    OptionDescriptor {
        id: "hide_own_swirls",
        label: "Hide your potion swirls",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(true),
        description: "In first person only. F5 and other players' swirls are unchanged.",
    },
    // The user's "particle physics toggle". "Particle physics"
    // means particles colliding with blocks; off skips the particles' collision checks, so they
    // pass through blocks ("no-clip"). The default here is on, which is vanilla 1.8.9.
    OptionDescriptor {
        id: "particle_physics",
        label: "Use particle physics",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(true),
        description: "Particles bump into blocks and settle on the ground, as in vanilla. Off \
                      skips their block collision checks: particles pass through floors and \
                      walls, and big bursts such as explosions cost less CPU.",
    },
    OptionDescriptor {
        id: "table",
        label: "Group values",
        kind: OptionKind::HiddenText { max_len: 160 },
        default: OptionValue::Text(Cow::Borrowed(DEFAULT_TABLE_TEXT)),
        description: "",
    },
];

/// The catalog entry.
pub static DESCRIPTOR: ModDescriptor = ModDescriptor {
    id: "particles",
    name: "Particles",
    category: Category::Pvp,
    description: "Scales the amount and size of specific particles: crits, potions, lava and more.",
    default_enabled: false,
    options: &OPTIONS,
    hud_elements: &[],
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_table_text_is_the_default_table_and_round_trips() {
        assert_eq!(parse_table(DEFAULT_TABLE_TEXT), DEFAULT_TABLE);
        let mut text = String::from("stale");
        write_table(&DEFAULT_TABLE, &mut text);
        assert_eq!(text, DEFAULT_TABLE_TEXT);
        // Group 0's row is what the sliders show on a fresh install.
        assert_eq!(
            GroupPercent::from_sliders(200.0, 100.0),
            DEFAULT_TABLE[0],
            "the slider defaults must match the crits row"
        );
        let mut extreme = DEFAULT_TABLE;
        extreme[6] = GroupPercent {
            amount: 1000,
            size: 300,
        };
        extreme[2] = GroupPercent {
            amount: 0,
            size: 25,
        };
        write_table(&extreme, &mut text);
        assert_eq!(parse_table(&text), extreme);
        let max_len = match DESCRIPTOR.option("table").map(|o| o.kind) {
            Some(OptionKind::HiddenText { max_len }) => max_len,
            other => panic!("table option kind {other:?}"),
        };
        let mut widest = String::new();
        write_table(
            &[GroupPercent {
                amount: 1000,
                size: 300,
            }; GROUP_COUNT],
            &mut widest,
        );
        assert!(widest.len() <= max_len, "{} > {max_len}", widest.len());
    }

    #[test]
    fn a_malformed_or_partial_table_keeps_defaults_and_clamps() {
        let table = parse_table("lava=300/150, explosion = 13/999 ,bogus=1/1,crit=,sharp=x/y,,");
        assert_eq!(
            table[4],
            GroupPercent {
                amount: 300,
                size: 150
            }
        );
        // Snapped to the 25% / 5% steps and clamped to the slider ranges.
        assert_eq!(
            table[6],
            GroupPercent {
                amount: 25,
                size: 300
            }
        );
        assert_eq!(table[0], DEFAULT_TABLE[0]);
        assert_eq!(table[1], DEFAULT_TABLE[1]);
        assert_eq!(table[2], DEFAULT_TABLE[2]);
        assert_eq!(parse_table(""), DEFAULT_TABLE);
        assert_eq!(parse_table("swirl=NaN/100"), DEFAULT_TABLE);
        assert_eq!(
            GroupPercent::from_sliders(-50.0, 1.0),
            GroupPercent {
                amount: 0,
                size: 25
            }
        );
        assert_eq!(
            GroupPercent::from_sliders(f32::NAN, f32::INFINITY).amount,
            100
        );
    }

    /// "Use particle physics" is a toggle that starts on, which is vanilla's collision.
    #[test]
    fn particle_physics_is_a_toggle_that_defaults_to_vanilla_collisions() {
        let option = DESCRIPTOR
            .option("particle_physics")
            .expect("the particle physics option");
        assert_eq!(option.kind, OptionKind::Toggle);
        assert_eq!(option.default, OptionValue::Bool(true));
        assert_eq!(option.label, "Use particle physics");
        // Shown after the hit toggles and before the hidden table.
        assert_eq!(
            DESCRIPTOR.option_index("particle_physics"),
            Some(DESCRIPTOR.option_index("hide_own_swirls").unwrap() + 1)
        );
    }

    #[test]
    fn the_group_choice_lists_every_table_group_in_key_order() {
        match DESCRIPTOR.option("group").map(|o| o.kind) {
            Some(OptionKind::Choice { choices }) => assert_eq!(choices, &GROUP_LABELS),
            other => panic!("group option kind {other:?}"),
        }
        assert_eq!(GROUP_KEYS.len(), GROUP_LABELS.len());
    }
}
