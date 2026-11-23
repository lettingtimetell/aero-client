//! The worst frames in big fights.
//!
//! The container sheets PvP games open within seconds (the player's own
//! inventory, whose sheet also holds the potion icons of the HUD and of the potion-status mod;
//! chests, which servers also use for their menus and shops; the crafting and enchanting tables;
//! the anvil; the hopper and the dispenser, which some server menus use) are loaded at install,
//! like the unicode pages and entity sheets the driver already prewarms, instead of in the frame
//! that first draws them. Each first draw read the PNG from the packs, decoded it and uploaded
//! 256 KB on the main thread: 1.1 ms in the frame the server's chest menu opened in a recorded
//! game, and 27-103 ms whenever the disk was busy. Same keys, same pixels: only earlier.
//!
//! `MC_W19_O2_GUI_PREWARM=0` (read once) leaves every sheet to load at its first draw, as before.

use std::sync::OnceLock;

use mc_render::Gpu;
use mc_ui::widget::{UiRenderer, UiTexture};

/// The sheets [`prewarm_gui_textures`] loads, as the paths `UiTexture::gui` keys them by.
pub(crate) const PREWARMED_GUI_TEXTURES: [&str; 7] = [
    "textures/gui/container/inventory.png",
    "textures/gui/container/generic_54.png",
    "textures/gui/container/crafting_table.png",
    "textures/gui/container/enchanting_table.png",
    "textures/gui/container/anvil.png",
    "textures/gui/container/hopper.png",
    "textures/gui/container/dispenser.png",
];

fn gui_prewarm_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("MC_W19_O2_GUI_PREWARM").map_or(true, |v| v != "0"))
}

/// Load [`PREWARMED_GUI_TEXTURES`] into the GUI renderer. A sheet the packs lack is remembered as
/// missing, exactly as at its first draw; a resource-pack reload clears the renderer's textures
/// and the driver calls this again right after.
pub(crate) fn prewarm_gui_textures(
    renderer: &mut UiRenderer,
    gpu: &Gpu,
    resources: &mc_assets::ResourceManager,
) {
    if !gui_prewarm_enabled() {
        return;
    }
    for path in PREWARMED_GUI_TEXTURES {
        let _ = renderer.ensure_texture(gpu, resources, &UiTexture::gui(path));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mc_ui::screens::container::ContainerArt;

    /// The prewarm only helps if its keys are exactly the keys the screens and the HUD ask for:
    /// every prewarmed sheet is one a container screen or the potion icons draw from, and each of
    /// those screens' sheets is prewarmed.
    #[test]
    fn the_prewarmed_sheets_are_the_keys_the_screens_and_the_hud_ask_for() {
        let screens = [
            ContainerArt::Player.texture(),
            ContainerArt::Chest { rows: 3 }.texture(),
            ContainerArt::Chest { rows: 6 }.texture(),
            ContainerArt::Crafting.texture(),
            ContainerArt::Enchanting(Default::default()).texture(),
            ContainerArt::Anvil(Default::default()).texture(),
            ContainerArt::Hopper.texture(),
            ContainerArt::Dispenser.texture(),
            UiTexture::gui(mc_ui::screens::hud::INVENTORY_TEXTURE),
        ];
        let prewarmed: Vec<UiTexture> = PREWARMED_GUI_TEXTURES
            .iter()
            .map(|path| UiTexture::gui(path))
            .collect();
        for key in &screens {
            assert!(prewarmed.contains(key), "{key:?} is not prewarmed");
        }
        for key in &prewarmed {
            assert!(
                screens.contains(key),
                "{key:?} is prewarmed but no screen draws it"
            );
        }
        // The comparison is sensitive: a sheet no listed screen uses is not among them.
        assert!(!screens.contains(&UiTexture::gui("textures/gui/container/furnace.png")));
    }
}
