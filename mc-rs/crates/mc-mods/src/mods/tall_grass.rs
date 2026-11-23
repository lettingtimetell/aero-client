//! Tall Grass (id `tall_grass`).
//!
//! Named after the vanilla feature, so it defaults **on**, which is vanilla. Off hides grass and
//! ferns, one and two blocks tall (not flowers, the shrub or dead bushes): they are not meshed,
//! and the F5 camera ray passes through them. Nothing else changes: hidden grass is still
//! targeted, still blocks melee hits behind it and still breaks, exactly as in vanilla, so no
//! packet differs. The block set is `mc_render::terrain::tall_grass`; the client half is
//! `mc-client/src/app/mods/w5_grass.rs`.

use crate::category::Category;
use crate::descriptor::ModDescriptor;

/// The catalog entry.
pub static DESCRIPTOR: ModDescriptor = ModDescriptor {
    id: "tall_grass",
    name: "Tall Grass",
    category: Category::World,
    description: "Grass and ferns, short and tall. Turn it off to hide them; the F5 camera then \
                  passes through them. Hidden grass still catches your crosshair and your hits, \
                  as in vanilla.",
    default_enabled: true,
    options: &[],
    hud_elements: &[],
};
