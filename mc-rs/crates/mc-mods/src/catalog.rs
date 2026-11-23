//! Every mod, in menu order (PLAN.md's table order).
//!
//! Adding a mod: create `src/mods/<id>.rs` with a `pub static DESCRIPTOR`, add `pub mod <id>;`
//! to `src/mods/mod.rs` and one line to [`ALL`] below. The tests at the bottom check the whole
//! catalog (unique ids, defaults that match their kinds, everything off by default).

use crate::category::Category;
use crate::descriptor::ModDescriptor;
use crate::mods;

/// The catalog. Within a window the menu lists mods in this order.
pub static ALL: [&ModDescriptor; 64] = [
    &mods::fps::DESCRIPTOR,
    &mods::cps::DESCRIPTOR,
    &mods::ping::DESCRIPTOR,
    &mods::coordinates::DESCRIPTOR,
    &mods::armor_status::DESCRIPTOR,
    &mods::potion_status::DESCRIPTOR,
    &mods::reach_display::DESCRIPTOR,
    &mods::item_info::DESCRIPTOR,
    &mods::keystrokes::DESCRIPTOR,
    &mods::spotify::DESCRIPTOR,
    &mods::old_animations::DESCRIPTOR,
    &mods::no_hurt_cam::DESCRIPTOR,
    &mods::hitboxes::DESCRIPTOR,
    &mods::stuck_arrows::DESCRIPTOR,
    &mods::shiny_pots::DESCRIPTOR,
    &mods::particles::DESCRIPTOR,
    &mods::fullbright::DESCRIPTOR,
    &mods::connected_textures::DESCRIPTOR,
    &mods::time_changer::DESCRIPTOR,
    &mods::sky::DESCRIPTOR,
    &mods::custom_sky::DESCRIPTOR,
    &mods::fog::DESCRIPTOR,
    &mods::vignette::DESCRIPTOR,
    &mods::tall_grass::DESCRIPTOR,
    &mods::fire_height::DESCRIPTOR,
    &mods::block_overlay::DESCRIPTOR,
    &mods::crosshair::DESCRIPTOR,
    &mods::motion_blur::DESCRIPTOR,
    &mods::color_grading::DESCRIPTOR,
    &mods::ambient_occlusion::DESCRIPTOR,
    &mods::anti_aliasing::DESCRIPTOR,
    &mods::anisotropic_filtering::DESCRIPTOR,
    &mods::volumetric_clouds::DESCRIPTOR,
    // The three shader packs.
    &mods::shaders_classic::DESCRIPTOR,
    &mods::shaders_dreamy::DESCRIPTOR,
    &mods::shaders_underwater::DESCRIPTOR,
    &mods::menu_blur::DESCRIPTOR,
    &mods::super_secret_settings::DESCRIPTOR,
    &mods::toggle_sprint::DESCRIPTOR,
    &mods::zoom::DESCRIPTOR,
    &mods::dynamic_fov::DESCRIPTOR,
    &mods::modern_keybinds::DESCRIPTOR,
    &mods::chat_macros::DESCRIPTOR,
    &mods::raw_input::DESCRIPTOR,
    &mods::borderless_fullscreen::DESCRIPTOR,
    // Replay.
    &mods::replay::DESCRIPTOR,
    &mods::boss_bar::DESCRIPTOR,
    &mods::scoreboard::DESCRIPTOR,
    &mods::chat::DESCRIPTOR,
    &mods::auto_gg::DESCRIPTOR,
    &mods::tab_list::DESCRIPTOR,
    &mods::titles::DESCRIPTOR,
    &mods::action_bar::DESCRIPTOR,
    &mods::scrollable_tooltips::DESCRIPTOR,
    &mods::centered_inventory::DESCRIPTOR,
    &mods::independent_gui::DESCRIPTOR,
    &mods::aero_title::DESCRIPTOR,
    &mods::smart_disconnect::DESCRIPTOR,
    &mods::waypoints::DESCRIPTOR,
    &mods::markers::DESCRIPTOR,
    &mods::nametags::DESCRIPTOR,
    &mods::skins_3d::DESCRIPTOR,
    &mods::cape_physics::DESCRIPTOR,
    // A low frame cap while the window is in the background.
    &mods::background_fps::DESCRIPTOR,
];

/// Every mod, in menu order.
#[must_use]
pub fn all() -> &'static [&'static ModDescriptor] {
    &ALL
}

/// The mod with this id.
#[must_use]
pub fn find(id: &str) -> Option<&'static ModDescriptor> {
    ALL.iter().copied().find(|descriptor| descriptor.id == id)
}

/// Position of the mod with this id in [`all`].
#[must_use]
pub fn index_of(id: &str) -> Option<usize> {
    ALL.iter().position(|descriptor| descriptor.id == id)
}

/// The mods listed in one window, in order, with their catalog index.
pub fn in_category(category: Category) -> impl Iterator<Item = (usize, &'static ModDescriptor)> {
    ALL.iter()
        .copied()
        .enumerate()
        .filter(move |(_, descriptor)| descriptor.category == category)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn ids_are_unique_and_snake_case() {
        let mut seen = HashSet::new();
        for descriptor in all() {
            assert!(
                seen.insert(descriptor.id),
                "duplicate mod id {}",
                descriptor.id
            );
            assert!(
                descriptor
                    .id
                    .chars()
                    .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_'),
                "{} is not snake_case",
                descriptor.id
            );
        }
    }

    /// These are ON by default because ON is vanilla or the established behaviour (raw mouse
    /// motion, F11 borderless, and the vanilla features named as switches: fog, the vignette, tall
    /// grass, and dynamic FOV, once called "No Dynamic FOV").
    /// The Aero title screen is the one non-vanilla mod that defaults on: it is the client's title
    /// screen, with the vanilla one kept as the off switch.
    /// Every other mod defaults off.
    ///
    /// `sky`, `stuck_arrows` and `modern_keybinds` joined the list later.
    /// 256-character chat remains on by default as an option inside Chat.
    /// `background_fps` changes nothing while the game has the focus: it only lowers the frame
    /// rate while the window is in the background.
    const DEFAULT_ON: [&str; 11] = [
        "raw_input",
        "borderless_fullscreen",
        "fog",
        "vignette",
        "tall_grass",
        "dynamic_fov",
        "aero_title",
        "sky",
        "stuck_arrows",
        "modern_keybinds",
        "background_fps",
    ];

    #[test]
    fn every_mod_is_off_by_default_except_the_ones_that_are_vanilla_or_todays_behaviour() {
        for descriptor in all() {
            assert_eq!(
                descriptor.default_enabled,
                DEFAULT_ON.contains(&descriptor.id),
                "{}: only {DEFAULT_ON:?} may default on",
                descriptor.id
            );
        }
    }

    #[test]
    fn every_option_default_matches_its_kind_and_range() {
        for descriptor in all() {
            let mut ids = HashSet::new();
            for option in descriptor.options {
                assert!(
                    ids.insert(option.id),
                    "{}: duplicate option {}",
                    descriptor.id,
                    option.id
                );
                assert!(
                    option.kind.accepts(&option.default),
                    "{}.{}: default {:?} does not match {:?}",
                    descriptor.id,
                    option.id,
                    option.default,
                    option.kind
                );
                assert_eq!(
                    option
                        .kind
                        .sanitize(option.default.clone(), &option.default),
                    option.default,
                    "{}.{}: default is outside its own range",
                    descriptor.id,
                    option.id
                );
            }
            let mut elements = HashSet::new();
            for element in descriptor.hud_elements {
                assert!(
                    elements.insert(element.id),
                    "{}: duplicate HUD element {}",
                    descriptor.id,
                    element.id
                );
                assert_eq!(
                    element.default_placement,
                    element.default_placement.sanitized(),
                    "{}.{}: default placement out of range",
                    descriptor.id,
                    element.id
                );
            }
        }
    }

    #[test]
    fn every_window_has_mods_and_lookup_works() {
        for category in Category::ALL {
            assert!(
                in_category(category).next().is_some(),
                "{category:?} would be an empty window"
            );
        }
        assert_eq!(find("fps").map(|d| d.name), Some("FPS"));
        assert_eq!(index_of("fps"), Some(0));
        assert!(find("nope").is_none());
    }
}
