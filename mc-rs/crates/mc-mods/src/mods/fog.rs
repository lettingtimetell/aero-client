//! Fog (id `fog`): vanilla's distance fog, **on by default** (on is vanilla). Keep the `id` stable (it is the settings-file key);
//! the `category` line must name the mod's home in `crate::menu_layout`, which decides where the
//! menu lists it (a test keeps the two in step).
//!
//! * **Off** removes the render-distance fog: `EntityRenderer.setupFog`'s air branch, in every
//!   dimension (the Nether's and the End's thick fog too), from terrain, entities, particles
//!   and clouds. Blindness fog always stays (removing it would be an unfair advantage), the
//!   in-cloud fog stays, and the sky keeps its horizon fade.
//! * **Clear water and lava** (default off) gives water and lava the air fog instead of their
//!   thick fog: with Fog on you see to the render distance under water, with Fog off there is
//!   no fog at all. It works whichever way the main switch is set.
//!
//! Renderer half: `mc_render::sky::fog::world_pass_fog_mode` (the one place the falloff
//! changes). Client half: `mc-client/src/app/mods/w4_fog_vignette.rs`.

use crate::category::Category;
use crate::descriptor::{ModDescriptor, OptionDescriptor, OptionKind, OptionValue};

/// Id of the "Clear water and lava" toggle.
pub const CLEAR_LIQUIDS: &str = "clear_liquids";

static OPTIONS: [OptionDescriptor; 1] = [OptionDescriptor {
    id: CLEAR_LIQUIDS,
    label: "Clear water and lava",
    kind: OptionKind::Toggle,
    default: OptionValue::Bool(false),
    description: "No thick fog under water or in lava. Works with Fog on or off.",
}];

/// The catalog entry.
pub static DESCRIPTOR: ModDescriptor = ModDescriptor {
    id: "fog",
    name: "Fog",
    category: Category::World,
    description: "Distance fog toward the edge of your render distance. Turn it off to see \
                  terrain and players sharp out to the edge. Blindness fog always stays. Clear \
                  water and lava works with Fog on or off.",
    default_enabled: true,
    options: &OPTIONS,
    hud_elements: &[],
};
