//! Vignette (id `vignette`): vanilla's darkened screen edges, **on by default** (on is
//! vanilla). Keep the `id` stable (it is
//! the settings-file key); the `category` line must name the mod's home in `crate::menu_layout`,
//! which decides where the menu lists it (a test keeps the two in step).
//!
//! Off skips `GuiIngame.renderVignette`'s brightness darkening (Fancy graphics only, strongest
//! in dark places). The red world-border warning, drawn through the same texture, still shows:
//! it is gameplay information. No options.
//!
//! Renderer half: `mc_render::post::OverlayPhase::ordinary_vignette`. Client half:
//! `mc-client/src/app/mods/w4_fog_vignette.rs`.

use crate::category::Category;
use crate::descriptor::ModDescriptor;

/// The catalog entry.
pub static DESCRIPTOR: ModDescriptor = ModDescriptor {
    id: "vignette",
    name: "Vignette",
    category: Category::Graphics,
    description: "The dark shading around the screen edges, strongest in dark places. Turn it \
                  off for a flat view; the red world-border warning still shows.",
    default_enabled: true,
    options: &[],
    hud_elements: &[],
};
