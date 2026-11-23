//! Where the mod menu lists every mod, with its one-line summary and search keywords.
//!
//! [`MOD_META`] holds each mod's menu wording, and a completeness test checks that every catalog
//! mod is placed, so the menu can never drift back into a catch-all "More" section.
//!
//! Nothing here is persisted: mod ids and `mc-rs-mods.json` are unchanged by the layout.

/// A category in window 1.
#[derive(Debug)]
pub struct MenuCategory {
    /// Stable id, used by the menu's state (never written to disk).
    pub id: &'static str,
    /// The name in window 1.
    pub title: &'static str,
    /// One line at the top of window 2. It is also the rule for placing future mods.
    pub description: &'static str,
    /// The sections of window 2, in order.
    pub sections: &'static [MenuSection],
}

/// A titled run of mods in window 2.
#[derive(Debug)]
pub struct MenuSection {
    /// The section title, drawn in capitals.
    pub title: &'static str,
    /// Mod ids, in order. A mod listed in a second category's "Also here" section keeps its home
    /// in the first category that lists it.
    pub mods: &'static [&'static str],
}

/// The menu's own text for one mod.
#[derive(Debug)]
pub struct ModMeta {
    /// The catalog id.
    pub id: &'static str,
    /// The line under the name in window 2: what the mod does, in at most 40 characters.
    pub summary: &'static str,
    /// Words people type for this mod, lower case. Search ignores spaces and accepts both
    /// spellings (color/colour), so keywords need not repeat either.
    pub keywords: &'static [&'static str],
}

/// The categories, in window-1 order.
pub static CATEGORIES: &[MenuCategory] = &[
    MenuCategory {
        id: "hud",
        title: "HUD",
        description: "Things drawn over the game while you play. Move them with Edit HUD.",
        sections: &[
            MenuSection {
                title: "Counters",
                mods: &["fps", "cps", "ping", "reach_display", "coordinates"],
            },
            MenuSection {
                title: "Gear",
                mods: &["armor_status", "potion_status", "item_info"],
            },
            MenuSection {
                title: "Overlays",
                mods: &["keystrokes", "spotify"],
            },
            MenuSection {
                title: "Vanilla HUD",
                mods: &[
                    "crosshair",
                    "scoreboard",
                    "boss_bar",
                    "titles",
                    "action_bar",
                ],
            },
        ],
    },
    MenuCategory {
        id: "pvp",
        title: "PvP",
        description: "How fights look and feel.",
        sections: &[
            MenuSection {
                title: "Feel",
                mods: &["old_animations", "no_hurt_cam"],
            },
            MenuSection {
                title: "Clarity",
                mods: &[
                    "hitboxes",
                    "particles",
                    "fire_height",
                    "stuck_arrows",
                    "shiny_pots",
                ],
            },
            MenuSection {
                title: "Also here",
                mods: &["crosshair"],
            },
        ],
    },
    MenuCategory {
        id: "world",
        title: "World",
        description: "What the world looks like: light, sky, clouds and blocks.",
        sections: &[
            MenuSection {
                title: "Light & time",
                mods: &["fullbright", "time_changer"],
            },
            MenuSection {
                title: "Sky & fog",
                mods: &["sky", "custom_sky", "volumetric_clouds", "fog"],
            },
            MenuSection {
                title: "Blocks",
                mods: &["block_overlay", "connected_textures", "tall_grass"],
            },
        ],
    },
    MenuCategory {
        id: "graphics",
        title: "Graphics",
        description: "Shader packs, image quality, and effects on the whole picture.",
        sections: &[
            MenuSection {
                title: "Shader packs",
                mods: &["shaders_classic", "shaders_dreamy", "shaders_underwater"],
            },
            MenuSection {
                title: "Quality",
                mods: &[
                    "anti_aliasing",
                    "anisotropic_filtering",
                    "ambient_occlusion",
                ],
            },
            MenuSection {
                title: "Effects",
                mods: &[
                    "motion_blur",
                    "color_grading",
                    "vignette",
                    "super_secret_settings",
                    "menu_blur",
                ],
            },
        ],
    },
    MenuCategory {
        id: "chat_menus",
        title: "Chat & Menus",
        description: "Chat, the tab list, tooltips, inventories and menus.",
        sections: &[
            MenuSection {
                title: "Chat",
                mods: &["chat", "chat_macros", "auto_gg"],
            },
            MenuSection {
                title: "Lists & tooltips",
                mods: &["tab_list", "scrollable_tooltips"],
            },
            MenuSection {
                title: "Inventory & menus",
                // Confirm Disconnect.
                mods: &[
                    "centered_inventory",
                    "independent_gui",
                    "aero_title",
                    "smart_disconnect",
                ],
            },
        ],
    },
    MenuCategory {
        id: "controls",
        title: "Controls",
        description: "Keys, mouse, camera and the game window.",
        sections: &[
            MenuSection {
                title: "Movement",
                mods: &["toggle_sprint", "modern_keybinds"],
            },
            MenuSection {
                title: "Camera",
                mods: &["zoom", "dynamic_fov"],
            },
            MenuSection {
                title: "Mouse & window",
                // Background FPS.
                mods: &["raw_input", "borderless_fullscreen", "background_fps"],
            },
            MenuSection {
                title: "Also here",
                mods: &["chat_macros"],
            },
        ],
    },
    MenuCategory {
        id: "players_places",
        title: "Players & Places",
        description: "Other players, and marking places in the world.",
        sections: &[
            MenuSection {
                title: "Players",
                mods: &["nametags", "skins_3d", "cape_physics"],
            },
            MenuSection {
                title: "Places",
                mods: &["waypoints", "markers"],
            },
        ],
    },
    MenuCategory {
        id: "replay",
        title: "Replay",
        description: "Record your games, save clips and export them to video.",
        sections: &[MenuSection {
            title: "Recording",
            mods: &["replay"],
        }],
    },
];

/// Summaries and keywords, one entry per catalog mod, in window order.
///
/// A summary says what the mod does in plain words (at most [`SUMMARY_MAX`] characters). The
/// keywords are the words people type that are not already in the name or summary: other names
/// people use for the feature ("blockhit", "f3+b"), the thing it changes ("gamma", "coords")
/// and common misnamings. A test checks that every catalog mod has exactly one entry.
pub static MOD_META: &[ModMeta] = &[
    // ---- HUD ----
    meta_row(
        "fps",
        "Frames per second",
        &[
            "framerate",
            "frame rate",
            "frame time",
            "ms",
            "counter",
            "performance",
        ],
    ),
    meta_row(
        "cps",
        "Clicks per second",
        &[
            "clicks",
            "click counter",
            "clicking",
            "jitter",
            "butterfly",
            "counter",
        ],
    ),
    meta_row(
        "ping",
        "Your latency to the server",
        &["latency", "lag", "ms", "connection", "counter"],
    ),
    meta_row(
        "reach_display",
        "How far away your last hit landed",
        &["reach", "distance", "range", "hit distance", "counter"],
    ),
    meta_row(
        "coordinates",
        "Your position and the way you face",
        &[
            "coords",
            "xyz",
            "position",
            "location",
            "f3",
            "direction",
            "facing",
            "compass",
        ],
    ),
    meta_row(
        "armor_status",
        "Armor and held item durability",
        &[
            "armor hud",
            "durability",
            "gear",
            "helmet",
            "chestplate",
            "leggings",
            "boots",
            "arrow count",
        ],
    ),
    meta_row(
        "potion_status",
        "Your potion effects and the time left",
        &[
            "potion",
            "potions",
            "effects",
            "potion effects",
            "buffs",
            "status",
            "timer",
        ],
    ),
    meta_row(
        "item_info",
        "Your held item's enchantments and damage",
        &[
            "enchant",
            "enchants",
            "enchantments",
            "damage",
            "item info",
            "sharpness",
            "protection",
        ],
    ),
    meta_row(
        "keystrokes",
        "WASD and mouse clicks as you press them",
        &["keys", "wasd", "key display", "cps", "keystroke", "inputs"],
    ),
    meta_row(
        "spotify",
        "Now playing, with cover and controls",
        &[
            // The mod's name until 2026-11-22, and the player it reads by default.
            "spotify",
            "music",
            "song",
            "now playing",
            "media",
            "player",
            "play",
            "pause",
            "skip",
            "track",
            "album",
        ],
    ),
    meta_row(
        "crosshair",
        "Shape, size and color of the crosshair",
        &["reticle", "cursor", "aim", "dot", "custom crosshair"],
    ),
    meta_row(
        "scoreboard",
        "Restyle, hide or move the sidebar",
        &["sidebar", "scores", "board", "score"],
    ),
    meta_row(
        "boss_bar",
        "Hide or move the boss bar",
        &[
            "bossbar",
            "boss health",
            "wither",
            "ender dragon",
            "adverts",
            "ads",
        ],
    ),
    meta_row(
        "titles",
        "Resize, hide or move the big titles",
        &[
            "title",
            "subtitle",
            "subtitles",
            "big text",
            "announcements",
        ],
    ),
    meta_row(
        "action_bar",
        "Hide or move the text above the hotbar",
        &["actionbar", "overlay message", "hotbar text", "now playing"],
    ),
    // ---- PvP ----
    meta_row(
        "old_animations",
        "1.7 block-hit, item poses and red armor",
        &[
            "1.7",
            "old animations",
            "blockhit",
            "block hit",
            "animations",
            "sword block",
            "red armor",
        ],
    ),
    meta_row(
        "no_hurt_cam",
        "No camera shake when you take damage",
        &[
            "hurtcam",
            "hurt camera",
            "damage shake",
            "screen shake",
            "camera shake",
            "tilt",
        ],
    ),
    meta_row(
        "hitboxes",
        "Outlines around players, mobs and items",
        &["hitbox", "f3+b", "bounding box", "boxes", "hit box"],
    ),
    meta_row(
        "particles",
        "Amount and size of each particle type",
        &[
            "crits",
            "crit",
            "crit particles",
            "sharpness",
            "particle multiplier",
            "potion",
            "splash",
            "lava",
            "explosion",
            "block breaking",
        ],
    ),
    meta_row(
        "fire_height",
        "Lower or fade the fire on your screen",
        &["low fire", "burning", "fire overlay", "flames"],
    ),
    meta_row(
        "stuck_arrows",
        "Arrows stuck in players",
        &["arrows", "arrow", "stuck", "invisible"],
    ),
    meta_row(
        "shiny_pots",
        "Potions glow with the 1.7 glint",
        &[
            "potion",
            "potions",
            "glint",
            "shiny",
            "enchant glint",
            "1.7",
        ],
    ),
    // ---- World ----
    meta_row(
        "fullbright",
        "See in the dark",
        &[
            "gamma",
            "brightness",
            "night vision",
            "bright",
            "light",
            "dark",
        ],
    ),
    meta_row(
        "time_changer",
        "Pick the time of day you see",
        &[
            "time", "day", "night", "noon", "midnight", "sunset", "clock", "daytime",
        ],
    ),
    meta_row(
        "sky",
        "The vanilla sky, sun, moon and stars",
        &["sun", "moon", "stars", "sunrise", "sunset"],
    ),
    meta_row(
        "custom_sky",
        "Your resource pack's custom sky",
        &["skybox", "sky pack", "resource pack", "texture pack"],
    ),
    meta_row(
        "volumetric_clouds",
        "3D clouds with real light and shadow",
        &[
            "clouds",
            "cloud",
            "3d clouds",
            "volumetric",
            "realistic clouds",
            "weather",
        ],
    ),
    meta_row(
        "fog",
        "Distance fog at the edge of your view",
        &[
            "render distance",
            "haze",
            "water fog",
            "lava fog",
            "mist",
            "no fog",
        ],
    ),
    meta_row(
        "block_overlay",
        "Outline of the block you look at",
        &[
            "block outline",
            "selection box",
            "outline",
            "highlight",
            "block highlight",
            "selection",
        ],
    ),
    meta_row(
        "connected_textures",
        "Seamless glass from your resource pack",
        &[
            "ctm",
            "connected glass",
            "glass",
            "seamless",
            "framed glass",
        ],
    ),
    meta_row(
        "tall_grass",
        "Show or hide grass and ferns",
        &[
            "grass",
            "ferns",
            "plants",
            "foliage",
            "hide grass",
            "bushes",
        ],
    ),
    // ---- Graphics ----
    meta_row(
        "shaders_classic",
        "Sun shadows, warm light and shiny water",
        &[
            "shader",
            "shaders",
            "shaderpack",
            "shader pack",
            "shadows",
            "bloom",
            "god rays",
            "light rays",
            "reflections",
            "waving plants",
            "realistic",
        ],
    ),
    meta_row(
        "shaders_dreamy",
        "Moody colors, soft glow and ground fog",
        &[
            "shader",
            "shaders",
            "shaderpack",
            "shader pack",
            "dreamy",
            "moody",
            "atmospheric",
            "ground fog",
            "mist",
            "lut",
            "color grading",
            "glow",
            "haze",
            "vaporwave",
            "pastel",
            "aesthetic",
            "cinematic",
        ],
    ),
    meta_row(
        "shaders_underwater",
        "The world as if under the sea",
        &[
            "shader",
            "shaders",
            "shaderpack",
            "shader pack",
            "underwater",
            "ocean",
            "sea",
            "caustics",
            "water",
            "aquatic",
            "marine",
        ],
    ),
    meta_row(
        "anti_aliasing",
        "Smooths jagged edges",
        &[
            "aa",
            "taa",
            "fxaa",
            "antialiasing",
            "jaggies",
            "jagged",
            "smoothing",
            "msaa",
            "smooth edges",
        ],
    ),
    meta_row(
        "anisotropic_filtering",
        "Sharper textures at shallow angles",
        &[
            "af",
            "aniso",
            "anisotropy",
            "texture filtering",
            "blurry textures",
            "mipmap",
        ],
    ),
    meta_row(
        "ambient_occlusion",
        "Soft shading in corners and crevices",
        &[
            "ao",
            "ssao",
            "gtao",
            "shading",
            "occlusion",
            "smooth lighting",
        ],
    ),
    meta_row(
        "motion_blur",
        "Blurs the picture as things move",
        &["blur", "motion", "trail", "smear"],
    ),
    meta_row(
        "color_grading",
        "Tune contrast, color and saturation",
        &[
            "grading",
            "cdl",
            "saturation",
            "contrast",
            "filter",
            "tint",
            "look",
        ],
    ),
    meta_row(
        "vignette",
        "Dark shading at the screen edges",
        &["dark edges", "edges", "screen edges", "corners", "shadow"],
    ),
    meta_row(
        "super_secret_settings",
        "Vanilla's hidden post-processing effects",
        &[
            "super secret",
            "secret settings",
            "post shaders",
            "vanilla shaders",
            "post processing",
            "creeper",
            "notch",
            "pencil",
            "invert",
            "wobble",
            "ntsc",
        ],
    ),
    meta_row(
        "menu_blur",
        "Blurs the game behind menus",
        &[
            "blur",
            "background blur",
            "pause menu",
            "inventory blur",
            "gui blur",
        ],
    ),
    // ---- Chat & Menus ----
    meta_row(
        "chat",
        "Timestamps, stacking, copy and history",
        &[
            "timestamps",
            "stack",
            "spam",
            "copy",
            "history",
            "chat history",
            "compact chat",
            "messages",
            "scrollback",
        ],
    ),
    meta_row(
        "chat_macros",
        "Send a message or command with one key",
        &[
            "macro",
            "macros",
            "command",
            "hotkey",
            "keybind",
            "bind",
            "auto text",
            "quick chat",
            "/play",
            "requeue",
        ],
    ),
    meta_row(
        "auto_gg",
        "Says gg when a match ends",
        &["gg", "good game", "match end", "auto message"],
    ),
    meta_row(
        "tab_list",
        "Restyle the player list",
        &[
            "tab",
            "tablist",
            "player list",
            "players",
            "ping",
            "header",
            "footer",
        ],
    ),
    meta_row(
        "scrollable_tooltips",
        "Scroll tooltips that don't fit",
        &[
            "tooltip",
            "tooltips",
            "scroll",
            "lore",
            "long tooltips",
            "item tooltip",
        ],
    ),
    meta_row(
        "centered_inventory",
        "Keeps the inventory centered",
        &[
            "inventory",
            "center",
            "potion shift",
            "inventory shift",
            "gui",
        ],
    ),
    meta_row(
        "independent_gui",
        "Separate inventory and hotbar sizes",
        &[
            "gui scale",
            "inventory scale",
            "hotbar scale",
            "hotbar size",
            "inventory size",
            "scale",
            "gui size",
        ],
    ),
    meta_row(
        "aero_title",
        "Aero title and loading screens",
        &[
            "title screen",
            "main menu",
            "menu",
            "loading screen",
            "logo",
            "greenhouse",
            "start screen",
            "home screen",
            "splash",
        ],
    ),
    meta_row(
        "smart_disconnect",
        "Asks before you leave a server",
        &[
            "confirm disconnect",
            "confirmation",
            "are you sure",
            "accidental disconnect",
            "leave server",
            "quit",
            "pause menu",
            "escape menu",
            "safe disconnect",
        ],
    ),
    // ---- Controls ----
    meta_row(
        "toggle_sprint",
        "Sprint without holding the key",
        &[
            "toggle sneak",
            "auto sprint",
            "sprint",
            "sneak",
            "crouch",
            "shift",
        ],
    ),
    meta_row(
        "modern_keybinds",
        "Held keys keep working after menus",
        &["keybinds", "held keys", "wasd", "1.13", "movement"],
    ),
    meta_row(
        "zoom",
        "Hold a key to zoom in",
        &["magnify", "spyglass", "scope", "binoculars"],
    ),
    meta_row(
        "dynamic_fov",
        "FOV from sprinting, potions and flying",
        &[
            "fov",
            "field of view",
            "speed fov",
            "sprint fov",
            "slowness fov",
            "underwater fov",
            "bow zoom",
        ],
    ),
    meta_row(
        "raw_input",
        "Mouse look without Windows acceleration",
        &[
            "raw input",
            "mouse",
            "acceleration",
            "sensitivity",
            "enhance pointer precision",
            "mouse accel",
        ],
    ),
    meta_row(
        "borderless_fullscreen",
        "F11 as a borderless window",
        &[
            "fullscreen",
            "borderless",
            "windowed",
            "f11",
            "alt tab",
            "exclusive",
        ],
    ),
    meta_row(
        "background_fps",
        "Low FPS while tabbed out or minimized",
        &[
            "unfocused",
            "out of focus",
            "alt tabbed",
            "inactive",
            "afk",
            "idle",
            "cpu",
            "power",
            "battery",
            "fps limit",
            "frame limit",
            "fps cap",
            "save energy",
        ],
    ),
    // ---- Players & Places ----
    meta_row(
        "nametags",
        "Restyle the names above players",
        &[
            "nametag",
            "name tag",
            "names",
            "own name",
            "f5",
            "nameplate",
        ],
    ),
    meta_row(
        "skins_3d",
        "Skin layers drawn in 3D",
        &[
            "voxel skin",
            "skin",
            "skins",
            "hat",
            "jacket",
            "sleeves",
            "outer layer",
        ],
    ),
    meta_row(
        "cape_physics",
        "Capes that wave as you move",
        &["cape", "capes", "cloak", "waving cape", "physics"],
    ),
    meta_row(
        "waypoints",
        "Mark places and see them from anywhere",
        &[
            "waypoint",
            "beacon",
            "minimap",
            "death point",
            "death waypoint",
            "location",
        ],
    ),
    meta_row(
        "markers",
        "Ping a spot for a few minutes",
        &["ping", "mark", "spot", "callout", "marker", "point"],
    ),
    // ---- Replay ----
    meta_row(
        "replay",
        "Record games and export clips to video",
        &[
            "record",
            "recording",
            "replay mod",
            "replays",
            "clip",
            "clips",
            "video",
            "export",
            "render",
            "mp4",
            "highlight",
            "microphone",
            "mic",
            "playback",
            "viewer",
            "rewind",
            "screen recording",
        ],
    ),
];

/// The longest summary, in characters.
pub const SUMMARY_MAX: usize = 40;

/// Mods named after a vanilla feature: their switch is the feature, on by default, and on is the
/// normal game ("Part of vanilla Minecraft").
pub static VANILLA_SWITCHES: &[&str] = &[
    "fog",
    "vignette",
    "tall_grass",
    "dynamic_fov",
    "sky",
    "stuck_arrows",
];

/// The note window 3 shows under a default-on mod's name, `None` for the others.
#[must_use]
pub fn default_on_note(descriptor: &crate::ModDescriptor) -> Option<&'static str> {
    if !descriptor.default_enabled {
        None
    } else if VANILLA_SWITCHES.contains(&descriptor.id) {
        Some("Part of vanilla Minecraft: on is the normal game.")
    } else {
        Some("On by default in Aero.")
    }
}

/// Extra search words for single settings, where people's word is not in the label or the
/// description: `(mod id, option id, keywords)`. Lower case; search ignores spaces and spelling.
pub static OPTION_KEYWORDS: &[(&str, &str, &[&str])] = &[
    ("anti_aliasing", "sharpness", &["sharpen", "sharp"]),
    (
        "hitboxes",
        "width",
        &["thickness", "thick", "thicker", "thin"],
    ),
    (
        "block_overlay",
        "outline_width",
        &["thickness", "thick", "thicker", "thin"],
    ),
    (
        "chat",
        "background_opacity",
        &["transparent", "transparency", "clear", "see through"],
    ),
    (
        "time_changer",
        "time",
        &[
            "noon", "midnight", "day", "night", "morning", "sunset", "sunrise",
        ],
    ),
    ("zoom", "factor", &["magnification", "zoom level", "fov"]),
    ("nametags", "scale", &["size", "bigger", "smaller"]),
    ("crosshair", "length", &["size", "bigger", "smaller"]),
    (
        "particles",
        "amount",
        &["multiplier", "more", "fewer", "count"],
    ),
    ("fire_height", "height", &["low fire", "lower"]),
    // Modern Keybinds' "Eating stops mining" moved here, with its search words.
    (
        "old_animations",
        "use_stops_mining",
        &["eat", "eating", "drink", "dig"],
    ),
    ("fullbright", "brightness", &["gamma"]),
    ("tab_list", "ping", &["latency", "ms"]),
    ("spotify", "opacity", &["transparent", "transparency"]),
    (
        "scoreboard",
        "background_opacity",
        &["transparent", "transparency"],
    ),
    // Each FOV source has its own slider now.
    ("dynamic_fov", "sprinting", &["sprint fov"]),
    ("dynamic_fov", "speed_effect", &["speed fov"]),
    ("dynamic_fov", "slowness_effect", &["slowness fov"]),
    // Chat's "Symbol shadows".
    (
        "chat",
        "symbol_shadows",
        &["unicode", "emoji", "special characters", "hearts", "stars"],
    ),
];

const fn meta_row(
    id: &'static str,
    summary: &'static str,
    keywords: &'static [&'static str],
) -> ModMeta {
    ModMeta {
        id,
        summary,
        keywords,
    }
}

/// The menu text for `id`, if it has any.
#[must_use]
pub fn meta(id: &str) -> Option<&'static ModMeta> {
    MOD_META.iter().find(|meta| meta.id == id)
}

/// The category that is `id`'s home: the first one that lists it.
#[must_use]
pub fn home_category(id: &str) -> Option<&'static MenuCategory> {
    CATEGORIES.iter().find(|category| category.lists(id))
}

/// The category with this category id (`"hud"`, `"chat_menus"`...).
#[must_use]
pub fn category(id: &str) -> Option<&'static MenuCategory> {
    CATEGORIES.iter().find(|category| category.id == id)
}

/// Every placed mod once, in window order: each category's sections top to bottom, skipping a
/// mod's second ("Also here") listing. Search uses it to order equal results.
pub fn menu_order() -> impl Iterator<Item = &'static str> {
    CATEGORIES.iter().enumerate().flat_map(|(index, category)| {
        category
            .sections
            .iter()
            .flat_map(|section| section.mods.iter().copied())
            .filter(move |id| !CATEGORIES[..index].iter().any(|earlier| earlier.lists(id)))
    })
}

/// The extra search words of one setting ([`OPTION_KEYWORDS`]); empty for most settings.
#[must_use]
pub fn option_keywords(mod_id: &str, option_id: &str) -> &'static [&'static str] {
    OPTION_KEYWORDS
        .iter()
        .find(|(m, o, _)| *m == mod_id && *o == option_id)
        .map_or(&[], |(_, _, keywords)| *keywords)
}

impl MenuCategory {
    /// Whether any section of this category lists `mod_id`.
    #[must_use]
    pub fn lists(&self, mod_id: &str) -> bool {
        self.sections
            .iter()
            .any(|section| section.mods.contains(&mod_id))
    }

    /// The section that lists `mod_id`.
    #[must_use]
    pub fn section_of(&self, mod_id: &str) -> Option<&'static MenuSection> {
        self.sections
            .iter()
            .find(|section| section.mods.contains(&mod_id))
    }

    /// Whether this category lists `mod_id` as a second listing (its home is another window).
    #[must_use]
    pub fn lists_as_also_here(&self, mod_id: &str) -> bool {
        self.lists(mod_id) && home_category(mod_id).is_some_and(|home| home.id != self.id)
    }

    /// How many mods the category lists, second listings included.
    #[must_use]
    pub fn mod_count(&self) -> usize {
        self.sections.iter().map(|section| section.mods.len()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog;
    use std::collections::HashSet;

    /// Plan §4.3/§5.1 item 2: every catalog mod sits in a named section, so a new mod that is not
    /// placed fails here instead of landing in a catch-all "More" section.
    #[test]
    fn every_catalog_mod_is_placed_and_every_listed_id_is_real() {
        for descriptor in catalog::all() {
            assert!(
                home_category(descriptor.id).is_some(),
                "{} is not placed in menu_layout::CATEGORIES",
                descriptor.id
            );
        }
        for category in CATEGORIES {
            for section in category.sections {
                for id in section.mods {
                    assert!(
                        catalog::find(id).is_some(),
                        "{} › {} lists {id}, which is not in the catalog",
                        category.title,
                        section.title
                    );
                }
            }
        }
        let placed: HashSet<&str> = menu_order().collect();
        assert_eq!(placed.len(), catalog::all().len());
        assert_eq!(menu_order().count(), catalog::all().len(), "each mod once");
    }

    /// The catalog mods `categories` does not place.
    fn unplaced(categories: &[MenuCategory]) -> Vec<&'static str> {
        catalog::all()
            .iter()
            .map(|descriptor| descriptor.id)
            .filter(|id| !categories.iter().any(|category| category.lists(id)))
            .collect()
    }

    /// The completeness check catches a mod that is not placed (a negative control for the test
    /// above, kept as a test).
    #[test]
    fn a_layout_missing_a_mod_is_caught() {
        static BROKEN: &[MenuCategory] = &[MenuCategory {
            id: "hud",
            title: "HUD",
            description: "Only FPS.",
            sections: &[MenuSection {
                title: "Counters",
                mods: &["fps"],
            }],
        }];
        let missing = unplaced(BROKEN);
        assert!(missing.contains(&"cps") && missing.contains(&"replay"));
        assert!(!missing.contains(&"fps"));
        assert_eq!(missing.len(), catalog::all().len() - 1);
        assert!(unplaced(CATEGORIES).is_empty());
    }

    #[test]
    fn categories_and_sections_are_well_formed() {
        let mut ids = HashSet::new();
        for category in CATEGORIES {
            assert!(
                ids.insert(category.id),
                "duplicate category {}",
                category.id
            );
            assert!(!category.title.is_empty() && !category.sections.is_empty());
            assert!(
                category.description.ends_with('.'),
                "{}: a description is a sentence",
                category.title
            );
            let mut in_category = HashSet::new();
            for section in category.sections {
                assert!(
                    !section.mods.is_empty(),
                    "{} › {}",
                    category.title,
                    section.title
                );
                assert_ne!(section.title, "More", "no catch-all section");
                for id in section.mods {
                    assert!(
                        in_category.insert(*id),
                        "{id} is listed twice in {}",
                        category.title
                    );
                }
            }
        }
    }

    /// A mod listed in a second window is listed there under "Also here", and only twice.
    #[test]
    fn second_listings_are_also_here() {
        for category in CATEGORIES {
            for section in category.sections {
                for id in section.mods {
                    let home = home_category(id).expect("placed");
                    if home.id == category.id {
                        assert_ne!(section.title, "Also here", "{id} is at home in {}", home.id);
                    } else {
                        assert_eq!(
                            section.title, "Also here",
                            "{id}'s second listing in {}",
                            category.title
                        );
                        assert!(category.lists_as_also_here(id));
                    }
                }
            }
        }
        let listings = |id: &str| CATEGORIES.iter().filter(|c| c.lists(id)).count();
        for descriptor in catalog::all() {
            assert!(listings(descriptor.id) <= 2, "{}", descriptor.id);
        }
    }

    /// A descriptor's `category` names the same home as this table, so the two never disagree.
    #[test]
    fn every_descriptor_category_is_its_home() {
        for descriptor in catalog::all() {
            let home = home_category(descriptor.id).expect("placed");
            assert_eq!(
                descriptor.category.menu_id(),
                home.id,
                "{}: `category:` says {:?}, menu_layout puts it in {}",
                descriptor.id,
                descriptor.category,
                home.title
            );
        }
        for category in crate::Category::ALL {
            assert!(
                self::category(category.menu_id()).is_some(),
                "{category:?} has no menu category"
            );
        }
        assert_eq!(crate::Category::ALL.len(), CATEGORIES.len());
    }

    /// Plan §4.6 (D3): the seven moves.
    #[test]
    fn the_seven_moves_hold() {
        let home = |id: &str| home_category(id).map(|c| c.id);
        assert_eq!(home("crosshair"), Some("hud"));
        assert!(category("pvp").is_some_and(|pvp| pvp.lists_as_also_here("crosshair")));
        for id in ["scoreboard", "boss_bar", "titles", "action_bar"] {
            assert_eq!(home(id), Some("hud"), "{id}");
        }
        assert_eq!(home("volumetric_clouds"), Some("world"));
        assert_eq!(home("chat_macros"), Some("chat_menus"));
        assert!(category("controls").is_some_and(|c| c.lists_as_also_here("chat_macros")));
        assert!(catalog::find("sounds").is_none() && home("sounds").is_none());
    }

    #[test]
    fn every_mod_has_one_summary_and_its_keywords() {
        let mut seen = HashSet::new();
        for row in MOD_META {
            assert!(
                catalog::find(row.id).is_some(),
                "MOD_META has unknown {}",
                row.id
            );
            assert!(seen.insert(row.id), "MOD_META lists {} twice", row.id);
            let chars = row.summary.chars().count();
            assert!(
                (1..=SUMMARY_MAX).contains(&chars),
                "{}: summary is {chars} characters: {:?}",
                row.id,
                row.summary
            );
            assert!(!row.summary.ends_with('.'), "{}: no full stop", row.id);
            assert!(
                row.summary.starts_with(|c: char| !c.is_lowercase()),
                "{}: sentence case",
                row.id
            );
            for spelling in ["colour", "armour", "centre", "grey", "favourite"] {
                assert!(
                    !row.summary.to_lowercase().contains(spelling),
                    "{}: US spelling",
                    row.id
                );
            }
            let mut words = HashSet::new();
            for keyword in row.keywords {
                assert!(!keyword.trim().is_empty(), "{}: empty keyword", row.id);
                assert_eq!(
                    *keyword,
                    keyword.to_lowercase(),
                    "{}: keywords are lower case",
                    row.id
                );
                assert!(words.insert(*keyword), "{}: {keyword} twice", row.id);
            }
        }
        for descriptor in catalog::all() {
            assert!(
                meta(descriptor.id).is_some(),
                "{} has no MOD_META",
                descriptor.id
            );
        }
        assert_eq!(MOD_META.len(), catalog::all().len());
    }

    #[test]
    fn default_on_mods_say_why() {
        for id in VANILLA_SWITCHES {
            let descriptor = catalog::find(id).unwrap_or_else(|| panic!("{id}"));
            assert!(
                descriptor.default_enabled,
                "{id}: a vanilla switch is on by default"
            );
        }
        let note = |id: &str| default_on_note(catalog::find(id).unwrap());
        assert_eq!(
            note("fog"),
            Some("Part of vanilla Minecraft: on is the normal game.")
        );
        assert_eq!(note("aero_title"), Some("On by default in Aero."));
        assert_eq!(note("raw_input"), Some("On by default in Aero."));
        assert_eq!(note("fps"), None);
    }

    #[test]
    fn option_keywords_name_real_visible_settings() {
        let mut seen = HashSet::new();
        for (mod_id, option_id, keywords) in OPTION_KEYWORDS {
            let option = catalog::find(mod_id)
                .and_then(|d| d.option(option_id))
                .unwrap_or_else(|| panic!("{mod_id}.{option_id} does not exist"));
            assert!(option.kind.is_visible(), "{mod_id}.{option_id}");
            assert!(
                seen.insert((mod_id, option_id)),
                "{mod_id}.{option_id} twice"
            );
            assert!(!keywords.is_empty());
            for keyword in *keywords {
                assert_eq!(*keyword, keyword.to_lowercase());
            }
        }
        assert_eq!(
            option_keywords("anti_aliasing", "sharpness"),
            ["sharpen", "sharp"]
        );
        assert!(option_keywords("fps", "display").is_empty());
    }
}
