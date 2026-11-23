//! A mod's home category, as a tag on its descriptor.
//!
//! Six windows became seven so the ~57 mods spread evenly. "Visual" split
//! into **World** (what the world looks and sounds like) and **Graphics** (post effects and image
//! quality); "Social" became **Players** (other players, cosmetics, places).
//!
//! The menu now lays mods out from
//! [`crate::menu_layout::CATEGORIES`] (sections, summaries, the "Also here" second listings and the
//! Replay category), not from this enum. "Interface" is shown as **Chat & Menus** and "Players" as
//! **Players & Places**; the variant names and [`Category::id`] stay as they were.

/// The category a mod belongs to. [`Category::ALL`] keeps the old seven-window order, then Replay;
/// the menu's order is [`crate::menu_layout::CATEGORIES`].
///
/// Where the menu lists a mod is decided by [`crate::menu_layout::CATEGORIES`]; this field names
/// the same home ([`Category::menu_id`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Category {
    /// Things drawn over the game: readouts and the vanilla HUD (scoreboard, titles...).
    Hud,
    /// How fights look and feel.
    Pvp,
    /// What the world looks like: light, sky, clouds and blocks.
    World,
    /// Shader packs, image quality and effects on the whole picture.
    Graphics,
    /// Chat & Menus: chat, the tab list, tooltips, inventories and menus.
    Interface,
    /// Players & Places: other players, and marking places in the world.
    Players,
    /// Keys, mouse, camera and the game window.
    Controls,
    /// Recording games, clips and video export.
    Replay,
}

impl Category {
    /// Every category (the old seven windows' order, then Replay).
    pub const ALL: [Category; 8] = [
        Category::Hud,
        Category::Pvp,
        Category::World,
        Category::Graphics,
        Category::Interface,
        Category::Players,
        Category::Controls,
        Category::Replay,
    ];

    /// Position in [`Category::ALL`].
    #[must_use]
    pub const fn index(self) -> usize {
        match self {
            Category::Hud => 0,
            Category::Pvp => 1,
            Category::World => 2,
            Category::Graphics => 3,
            Category::Interface => 4,
            Category::Players => 5,
            Category::Controls => 6,
            Category::Replay => 7,
        }
    }

    /// Stable lower-case id.
    #[must_use]
    pub const fn id(self) -> &'static str {
        match self {
            Category::Hud => "hud",
            Category::Pvp => "pvp",
            Category::World => "world",
            Category::Graphics => "graphics",
            Category::Interface => "interface",
            Category::Players => "players",
            Category::Controls => "controls",
            Category::Replay => "replay",
        }
    }

    /// The category's name in the menu (the same as its [`crate::menu_layout::MenuCategory`]).
    #[must_use]
    pub const fn title(self) -> &'static str {
        match self {
            Category::Hud => "HUD",
            Category::Pvp => "PvP",
            Category::World => "World",
            Category::Graphics => "Graphics",
            Category::Interface => "Chat & Menus",
            Category::Players => "Players & Places",
            Category::Controls => "Controls",
            Category::Replay => "Replay",
        }
    }

    /// The id of the menu category that is this category's window
    /// ([`crate::menu_layout::category`]).
    #[must_use]
    pub const fn menu_id(self) -> &'static str {
        match self {
            Category::Hud => "hud",
            Category::Pvp => "pvp",
            Category::World => "world",
            Category::Graphics => "graphics",
            Category::Interface => "chat_menus",
            Category::Players => "players_places",
            Category::Controls => "controls",
            Category::Replay => "replay",
        }
    }

    /// The menu category's one-line description, the rule for placing future mods there.
    #[must_use]
    pub fn subtitle(self) -> &'static str {
        crate::menu_layout::category(self.menu_id()).map_or("", |category| category.description)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_category_names_its_menu_category() {
        for category in Category::ALL {
            let menu = crate::menu_layout::category(category.menu_id())
                .unwrap_or_else(|| panic!("{category:?} has no menu category"));
            assert_eq!(category.title(), menu.title);
            assert_eq!(category.subtitle(), menu.description);
            assert_eq!(Category::ALL[category.index()], category);
        }
    }
}
