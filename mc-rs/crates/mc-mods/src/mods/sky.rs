//! Sky (id `sky`): the vanilla sky, **on by default** (on is vanilla).
//!
//! * **Off** skips everything vanilla draws in the sky: the sky colour plane, the sunrise and
//!   sunset glow, the sun, the moon, the stars, the dark horizon and void planes, and the End's
//!   sky box. Left behind: the fog colour (the frame's clear colour) and a Custom Sky pack's
//!   layers, which then no longer have a vanilla sun, glow or sky colour showing through or on
//!   top of them.
//! * **Sun and moon** / **Stars** (both default on): drop just those while the vanilla sky stays,
//!   for a pack that draws its own sun or stars.
//!
//! Renderer half: `mc_render::sky::dome::SkyPhase::{vanilla_sky, vanilla_sun_and_moon,
//! vanilla_stars}`. Client half: `mc-client/src/app/mods/x7_world.rs`. Visual only: no packets.
//! Keep the `id` stable (it is the settings-file key).

use crate::category::Category;
use crate::descriptor::{ModDescriptor, OptionDescriptor, OptionKind, OptionValue};

/// Id of the "Sun and moon" toggle.
pub const SUN_AND_MOON: &str = "sun_and_moon";
/// Id of the "Stars" toggle.
pub const STARS: &str = "stars";

static OPTIONS: [OptionDescriptor; 2] = [
    OptionDescriptor {
        id: SUN_AND_MOON,
        label: "Sun and moon",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(true),
        description: "The vanilla sun and moon.",
    },
    OptionDescriptor {
        id: STARS,
        label: "Stars",
        kind: OptionKind::Toggle,
        default: OptionValue::Bool(true),
        description: "The vanilla stars at night.",
    },
];

/// The catalog entry.
pub static DESCRIPTOR: ModDescriptor = ModDescriptor {
    id: "sky",
    name: "Sky",
    category: Category::World,
    description: "The vanilla sky: sky color, sunrise glow, sun, moon and stars. Turn it off to \
                  leave only a Custom Sky pack's sky (or the plain fog color), with no vanilla \
                  sun or glow fighting it. Sun and moon and Stars hide just those.",
    default_enabled: true,
    options: &OPTIONS,
    hud_elements: &[],
};
