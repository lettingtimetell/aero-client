//! Rules for the mod menu's settings window: which
//! settings are hidden or dimmed while they do nothing, named groups of settings, and the special
//! layouts.
//!
//! **Rows for a mod go at that mod's own marker line**, so parallel changes never edit the same
//! line. Headings that follow from an
//! option's kind (KEYS for keybinds, STYLE for colours, shadows and backgrounds) are worked out by
//! the menu and need no row here.
//!
//! The logic for the settings window lives here too, so it is testable without the UI:
//! [`option_state`] (shown, dimmed or hidden right now, allocation-free) and [`settings_layout`]
//! (the blocks and headings of one mod's settings).

use crate::descriptor::{ModDescriptor, OptionDescriptor, OptionKind, OptionValue};
use crate::settings::ModSettings;

/// A condition on the same mod's settings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cond {
    /// The toggle with this id is on.
    On(&'static str),
    /// The toggle with this id is off.
    Off(&'static str),
    /// The choice with this id is at one of these indices.
    ChoiceIn(&'static str, &'static [usize]),
    /// The mod itself is switched on.
    ModOn,
}

/// What the settings window does with a setting whose condition does not hold.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unmet {
    /// Hide it: it belongs to another mode. A setting found by search is shown dimmed instead.
    Hide,
    /// Show it dimmed, with this reason on hover ("Needs Background on").
    Dim(&'static str),
}

/// `option` of `mod_id` does something only while `when` holds.
#[derive(Debug, PartialEq, Eq)]
pub struct OptionRule {
    /// The mod's catalog id.
    pub mod_id: &'static str,
    /// The setting the rule is about.
    pub option: &'static str,
    /// When the setting applies.
    pub when: Cond,
    /// What happens to it otherwise.
    pub unmet: Unmet,
}

/// A named heading over some of a mod's settings in window 3.
#[derive(Debug)]
pub struct OptionGroup {
    /// The mod's catalog id.
    pub mod_id: &'static str,
    /// The heading, drawn in capitals.
    pub title: &'static str,
    /// The settings under it, in order.
    pub options: &'static [&'static str],
    /// Whether the group starts collapsed.
    pub collapsed: bool,
}

/// Settings drawn in pairs on one row (Chat Macros: each macro's key beside its text).
#[derive(Debug)]
pub struct PairedRows {
    /// The mod's catalog id.
    pub mod_id: &'static str,
    /// `(left, right)` setting ids, one pair per row.
    pub pairs: &'static [(&'static str, &'static str)],
}

/// A choice drawn as a row of tabs above the settings it switches between (Particles).
#[derive(Debug)]
pub struct TabbedPicker {
    /// The mod's catalog id.
    pub mod_id: &'static str,
    /// The choice setting drawn as tabs.
    pub picker: &'static str,
    /// The settings whose meaning follows the picked tab.
    pub edits: &'static [&'static str],
}

/// Hide and dim rules.
pub static RULES: &[OptionRule] = &[
    // Settings of another mode are hidden.
    hide(
        "anti_aliasing",
        "sharpness",
        Cond::ChoiceIn("mode", &[crate::mods::anti_aliasing::MODE_TAA]),
    ),
    hide(
        "anti_aliasing",
        "texture_edges",
        Cond::ChoiceIn("mode", &[crate::mods::anti_aliasing::MODE_FXAA]),
    ),
    hide(
        "crosshair",
        "color",
        Cond::ChoiceIn("color_mode", &[crate::mods::crosshair::COLOR_MODE_CUSTOM]),
    ),
    // Settings that need a switch on are dimmed. The readouts' box colour needs the box.
    dim(
        "fps",
        "background_color",
        Cond::On("background"),
        NEEDS_BACKGROUND,
    ),
    dim(
        "cps",
        "background_color",
        Cond::On("background"),
        NEEDS_BACKGROUND,
    ),
    dim(
        "ping",
        "background_color",
        Cond::On("background"),
        NEEDS_BACKGROUND,
    ),
    dim(
        "coordinates",
        "background_color",
        Cond::On("background"),
        NEEDS_BACKGROUND,
    ),
    dim(
        "reach_display",
        "background_color",
        Cond::On("background"),
        NEEDS_BACKGROUND,
    ),
    dim(
        "potion_status",
        "background_color",
        Cond::On("background"),
        NEEDS_BACKGROUND,
    ),
    // Toggle Sprint's style is the status text's style.
    dim(
        "toggle_sprint",
        "text_color",
        Cond::On("show_text"),
        NEEDS_STATUS_TEXT,
    ),
    dim(
        "toggle_sprint",
        "text_shadow",
        Cond::On("show_text"),
        NEEDS_STATUS_TEXT,
    ),
    dim(
        "toggle_sprint",
        "background",
        Cond::On("show_text"),
        NEEDS_STATUS_TEXT,
    ),
    dim(
        "toggle_sprint",
        "background_color",
        Cond::On("show_text"),
        NEEDS_STATUS_TEXT,
    ),
    dim(
        "toggle_sprint",
        "background_color",
        Cond::On("background"),
        NEEDS_BACKGROUND,
    ),
    dim(
        "toggle_sprint",
        "brackets",
        Cond::On("show_text"),
        NEEDS_STATUS_TEXT,
    ),
    dim(
        "potion_status",
        "blink_time_only",
        Cond::On("blink"),
        "Needs Blink when ending on",
    ),
    dim(
        "potion_status",
        "blink_seconds",
        Cond::On("blink"),
        "Needs Blink when ending on",
    ),
    // Durability "Off" draws no number to colour.
    dim(
        "armor_status",
        "color_by_durability",
        Cond::ChoiceIn("durability", &[0, 1, 2]),
        "Needs a Durability number",
    ),
    dim(
        "keystrokes",
        "show_cps",
        Cond::On("show_mouse"),
        "Needs Mouse buttons on",
    ),
    dim(
        "skins_3d",
        "first_person",
        Cond::On("sleeves"),
        "Needs Sleeves on",
    ),
    // Hide switches: the rest of the mod draws nothing while they are on.
    dim(
        "scoreboard",
        "hide_numbers",
        Cond::Off("hide"),
        NEEDS_SCOREBOARD,
    ),
    dim(
        "scoreboard",
        "background_opacity",
        Cond::Off("hide"),
        NEEDS_SCOREBOARD,
    ),
    dim(
        "scoreboard",
        "text_shadow",
        Cond::Off("hide"),
        NEEDS_SCOREBOARD,
    ),
    dim("boss_bar", "show_text", Cond::Off("hide"), NEEDS_BOSS_BAR),
    dim("boss_bar", "show_bar", Cond::Off("hide"), NEEDS_BOSS_BAR),
    dim("boss_bar", "text_shadow", Cond::Off("hide"), NEEDS_BOSS_BAR),
    dim(
        "boss_bar",
        "text_shadow",
        Cond::On("show_text"),
        "Needs Show name on",
    ),
    dim("titles", "title_scale", Cond::Off("hide"), NEEDS_TITLES),
    dim("titles", "subtitle_scale", Cond::Off("hide"), NEEDS_TITLES),
    dim("titles", "text_shadow", Cond::Off("hide"), NEEDS_TITLES),
    dim(
        "action_bar",
        "text_shadow",
        Cond::Off("hide"),
        "Needs Hide action bar off",
    ),
    // Volumetric Clouds (B5): the distance belongs to the "Above you" height mode only.
    hide(
        "volumetric_clouds",
        "distance_above",
        Cond::ChoiceIn("height", &[0]),
    ),
    // Classic Shaders: every effect's settings need its switch on.
    dim(
        "shaders_classic",
        "sun_strength",
        Cond::On("sun_shading"),
        "Needs Sun shading on",
    ),
    dim(
        "shaders_classic",
        "shade_brightness",
        Cond::On("sun_shading"),
        "Needs Sun shading on",
    ),
    dim(
        "shaders_classic",
        "sunset_warmth",
        Cond::On("sun_shading"),
        "Needs Sun shading on",
    ),
    dim(
        "shaders_classic",
        "moonlight_tint",
        Cond::On("sun_shading"),
        "Needs Sun shading on",
    ),
    dim(
        "shaders_classic",
        "shadow_distance",
        Cond::On("shadows"),
        "Needs Shadows on",
    ),
    dim(
        "shaders_classic",
        "shadow_quality",
        Cond::On("shadows"),
        "Needs Shadows on",
    ),
    dim(
        "shaders_classic",
        "shadow_softness",
        Cond::On("shadows"),
        "Needs Shadows on",
    ),
    dim(
        "shaders_classic",
        "shadow_strength",
        Cond::On("shadows"),
        "Needs Shadows on",
    ),
    dim(
        "shaders_classic",
        "plant_shadows",
        Cond::On("shadows"),
        "Needs Shadows on",
    ),
    dim(
        "shaders_classic",
        "player_shadows",
        Cond::On("shadows"),
        "Needs Shadows on",
    ),
    dim(
        "shaders_classic",
        "water_reflections",
        Cond::On("water"),
        "Needs Shader water on",
    ),
    dim(
        "shaders_classic",
        "water_waves",
        Cond::On("water"),
        "Needs Shader water on",
    ),
    dim(
        "shaders_classic",
        "water_clarity",
        Cond::On("water"),
        "Needs Shader water on",
    ),
    dim(
        "shaders_classic",
        "light_rays_strength",
        Cond::On("light_rays"),
        "Needs Light rays on",
    ),
    dim(
        "shaders_classic",
        "bloom_strength",
        Cond::On("bloom"),
        "Needs Bloom on",
    ),
    // Dreamy Shaders: each fog height mode shows its own setting, the custom fog color only in
    // Custom; the fog and light-ray settings need their switch on.
    hide(
        "shaders_dreamy",
        "fog_depth",
        Cond::ChoiceIn("fog_height_mode", &[0]),
    ),
    hide(
        "shaders_dreamy",
        "fog_level",
        Cond::ChoiceIn("fog_height_mode", &[1]),
    ),
    hide(
        "shaders_dreamy",
        "fog_color",
        Cond::ChoiceIn("fog_color_mode", &[1]),
    ),
    dim(
        "shaders_dreamy",
        "fog_height_mode",
        Cond::On("ground_fog"),
        "Needs Ground fog on",
    ),
    dim(
        "shaders_dreamy",
        "fog_depth",
        Cond::On("ground_fog"),
        "Needs Ground fog on",
    ),
    dim(
        "shaders_dreamy",
        "fog_level",
        Cond::On("ground_fog"),
        "Needs Ground fog on",
    ),
    dim(
        "shaders_dreamy",
        "fog_density",
        Cond::On("ground_fog"),
        "Needs Ground fog on",
    ),
    dim(
        "shaders_dreamy",
        "fog_softness",
        Cond::On("ground_fog"),
        "Needs Ground fog on",
    ),
    dim(
        "shaders_dreamy",
        "fog_color_mode",
        Cond::On("ground_fog"),
        "Needs Ground fog on",
    ),
    dim(
        "shaders_dreamy",
        "fog_color",
        Cond::On("ground_fog"),
        "Needs Ground fog on",
    ),
    dim(
        "shaders_dreamy",
        "fog_sun_glow",
        Cond::On("ground_fog"),
        "Needs Ground fog on",
    ),
    dim(
        "shaders_dreamy",
        "fog_wind",
        Cond::On("ground_fog"),
        "Needs Ground fog on",
    ),
    dim(
        "shaders_dreamy",
        "fog_start",
        Cond::On("ground_fog"),
        "Needs Ground fog on",
    ),
    dim(
        "shaders_dreamy",
        "fog_quality",
        Cond::On("ground_fog"),
        "Needs Ground fog on",
    ),
    dim(
        "shaders_dreamy",
        "shadow_strength",
        Cond::On("soft_shadows"),
        "Needs Soft shadows on",
    ),
    dim(
        "shaders_dreamy",
        "rays_strength",
        Cond::On("light_rays"),
        "Needs Light rays on",
    ),
    dim(
        "shaders_dreamy",
        "rays_length",
        Cond::On("light_rays"),
        "Needs Light rays on",
    ),
    OptionRule {
        mod_id: "shaders_underwater",
        option: "clarity",
        when: Cond::ChoiceIn("mode", &[0]),
        unmet: Unmet::Hide,
    },
    OptionRule {
        mod_id: "shaders_underwater",
        option: "caustic_strength",
        when: Cond::On("caustics"),
        unmet: Unmet::Dim("Needs Caustics on"),
    },
    OptionRule {
        mod_id: "shaders_underwater",
        option: "caustic_size",
        when: Cond::On("caustics"),
        unmet: Unmet::Dim("Needs Caustics on"),
    },
    OptionRule {
        mod_id: "shaders_underwater",
        option: "shaft_strength",
        when: Cond::On("light_shafts"),
        unmet: Unmet::Dim("Needs Light shafts on"),
    },
    OptionRule {
        mod_id: "shaders_underwater",
        option: "quality",
        when: Cond::On("light_shafts"),
        unmet: Unmet::Dim("Needs Light shafts on"),
    },
    OptionRule {
        mod_id: "shaders_underwater",
        option: "snow_amount",
        when: Cond::On("marine_snow"),
        unmet: Unmet::Dim("Needs Marine snow on"),
    },
    OptionRule {
        mod_id: "shaders_underwater",
        option: "refraction_strength",
        when: Cond::On("refraction"),
        unmet: Unmet::Dim("Needs Refraction wobble on"),
    },
    // Replay: the microphone's volume does nothing while the microphone is not recorded.
    dim(
        "replay",
        "mic_volume",
        Cond::On("mic"),
        "Needs Record microphone on",
    ),
    // (the minimum loading time's hide rule; that option was retired)
    // Super Secret Settings: the ASCII style settings belong to the ASCII choice only.
    OptionRule {
        mod_id: "super_secret_settings",
        option: "ascii_size",
        when: Cond::ChoiceIn(
            "shader",
            &[crate::mods::super_secret_settings::ASCII_CHOICE],
        ),
        unmet: Unmet::Hide,
    },
    OptionRule {
        mod_id: "super_secret_settings",
        option: "ascii_colors",
        when: Cond::ChoiceIn(
            "shader",
            &[crate::mods::super_secret_settings::ASCII_CHOICE],
        ),
        unmet: Unmet::Hide,
    },
    // Replay: the source only matters while desktop audio is recorded (its volume still applies
    // to replays recorded earlier, so it stays live).
    dim(
        "replay",
        "desktop_source",
        Cond::On("desktop_audio"),
        "Needs Record desktop audio on",
    ),
];

/// Named groups.
pub static GROUPS: &[OptionGroup] = &[
    group(
        "old_animations",
        "First person",
        &[
            "blockhit",
            "punch_during_use",
            "item_positions",
            "bow",
            "rod",
            "translucent_held",
            "use_stops_mining",
        ],
    ),
    group(
        "old_animations",
        "Other players",
        &["sword_block_3p", "armor_red"],
    ),
    group(
        "old_animations",
        "Camera and hearts",
        &["smooth_sneak", "old_health"],
    ),
    group(
        "hitboxes",
        "Show for",
        &["players", "mobs", "items", "projectiles", "other"],
    ),
    group(
        "hitboxes",
        "Lines",
        &["color", "width", "eye_line", "look_vector"],
    ),
    group("crosshair", "Color", &["color_mode", "color"]),
    group(
        "block_overlay",
        "Outline",
        &["outline_color", "outline_width", "adventure_outline"],
    ),
    group("block_overlay", "Fill", &["fill_color"]),
    group(
        "potion_status",
        "Blinking",
        &["blink", "blink_time_only", "blink_seconds"],
    ),
    group(
        "zoom",
        "While zoomed",
        &["scroll", "smooth_camera", "sensitivity"],
    ),
    group(
        "skins_3d",
        "Parts",
        &["hat", "jacket", "sleeves", "pants", "first_person"],
    ),
    OptionGroup {
        mod_id: "color_grading",
        title: "Per channel",
        options: &[
            "slope_r", "slope_g", "slope_b", "offset_r", "offset_g", "offset_b", "power_r",
            "power_g", "power_b",
        ],
        collapsed: true,
    },
    // Chat (B1): the two history settings, which work with the Chat mod off.
    group("chat", "History", &["keep_chat", "history_length"]),
    group(
        "shaders_classic",
        "Lighting",
        &[
            "sun_shading",
            "sun_strength",
            "shade_brightness",
            "sun_path_tilt",
            "sunset_warmth",
            "moonlight_tint",
        ],
    ),
    group(
        "shaders_classic",
        "Shadows",
        &[
            "shadows",
            "shadow_distance",
            "shadow_quality",
            "shadow_softness",
            "shadow_strength",
            "plant_shadows",
            "player_shadows",
        ],
    ),
    group(
        "shaders_classic",
        "Water",
        &["water", "water_reflections", "water_waves", "water_clarity"],
    ),
    group(
        "shaders_classic",
        "Plants",
        &["waving_plants", "waving_leaves", "waving_strength"],
    ),
    group(
        "shaders_classic",
        "Glow",
        &[
            "light_rays",
            "light_rays_strength",
            "sun_glow",
            "bloom",
            "bloom_strength",
        ],
    ),
    group(
        "shaders_classic",
        "Color",
        &[
            "tone_curve",
            "exposure",
            "contrast",
            "saturation",
            "vibrance",
        ],
    ),
    group("shaders_dreamy", "Look", &["preset"]),
    group(
        "shaders_dreamy",
        "Color",
        &[
            "exposure",
            "contrast",
            "saturation",
            "vibrance",
            "temperature",
            "tint",
            "hue_shift",
        ],
    ),
    OptionGroup {
        mod_id: "shaders_dreamy",
        title: "Color wheels",
        options: &[
            "shadows_hue",
            "shadows_strength",
            "midtones_hue",
            "midtones_strength",
            "highlights_hue",
            "highlights_strength",
        ],
        collapsed: true,
    },
    OptionGroup {
        mod_id: "shaders_dreamy",
        title: "Split toning",
        options: &[
            "split_shadows",
            "split_highlights",
            "split_balance",
            "split_strength",
        ],
        collapsed: true,
    },
    OptionGroup {
        mod_id: "shaders_dreamy",
        title: "Fade and gradient map",
        options: &[
            "fade",
            "gradient_mix",
            "gradient_dark",
            "gradient_mid",
            "gradient_light",
        ],
        collapsed: true,
    },
    group(
        "shaders_dreamy",
        "Glow and sky",
        &[
            "glow",
            "glow_radius",
            "halation",
            "sky_tint",
            "sky_tint_strength",
            "haze",
            "haze_start",
        ],
    ),
    group(
        "shaders_dreamy",
        "Ground fog",
        &[
            "ground_fog",
            "fog_height_mode",
            "fog_depth",
            "fog_level",
            "fog_density",
            "fog_softness",
            "fog_color_mode",
            "fog_color",
            "fog_sun_glow",
            "fog_wind",
            "fog_start",
            "fog_quality",
        ],
    ),
    group(
        "shaders_dreamy",
        "Lighting",
        &[
            "sunlight",
            "soft_shadows",
            "shadow_strength",
            "light_rays",
            "rays_strength",
            "rays_length",
        ],
    ),
    group(
        "shaders_dreamy",
        "Lens",
        &["vignette", "grain", "chromatic_aberration"],
    ),
    OptionGroup {
        mod_id: "shaders_underwater",
        title: "General",
        options: &["mode", "preset"],
        collapsed: false,
    },
    OptionGroup {
        mod_id: "shaders_underwater",
        title: "Water",
        options: &[
            "water_color",
            "clarity",
            "clear_range",
            "red_fade",
            "depth_darkening",
            "surface_sky",
        ],
        collapsed: false,
    },
    OptionGroup {
        mod_id: "shaders_underwater",
        title: "Light",
        options: &[
            "caustics",
            "caustic_strength",
            "caustic_size",
            "wave_speed",
            "light_shafts",
            "shaft_strength",
        ],
        collapsed: false,
    },
    OptionGroup {
        mod_id: "shaders_underwater",
        title: "Effects",
        options: &[
            "marine_snow",
            "snow_amount",
            "refraction",
            "refraction_strength",
        ],
        collapsed: false,
    },
    OptionGroup {
        mod_id: "shaders_underwater",
        title: "Under water",
        options: &["underwater_clarity", "vanilla_overlay"],
        collapsed: true,
    },
    OptionGroup {
        mod_id: "shaders_underwater",
        title: "Performance",
        options: &["quality"],
        collapsed: true,
    },
    // Replay: "Replay viewer" stays on top (ungrouped); the four keys go under KEYS.
    group(
        "replay",
        "Recording",
        &["auto_start", "indicator", "mic", "mic_volume"],
    ),
    group(
        "replay",
        "Clips",
        &[
            "clip_length",
            "clip_max_size",
            "clip_auto_export",
            "clip_fps",
            "clip_resolution",
        ],
    ),
    OptionGroup {
        mod_id: "replay",
        title: "Export defaults",
        options: &[
            "export_fps",
            "export_resolution",
            "export_quality",
            "export_codec",
            "export_motion_blur",
            "export_audio",
            "export_mic",
        ],
        collapsed: true,
    },
    group("replay", "Storage", &["max_storage_gb"]),
    // The retired "Loading screen" group stood here; the "Main menu" group takes
    // its place, so it stays the first heading on the mod's page (headings follow this list's
    // order).
    group("aero_title", "Main menu", &["menu_scene"]),
    group("aero_title", "Logo", &["logo_style"]),
    // (the "Main menu" group is at above)
    // Dynamic FOV: the movement sliders under their master Strength, then the other sources.
    group(
        "dynamic_fov",
        "Movement",
        &[
            "strength",
            "sprinting",
            "speed_effect",
            "slowness_effect",
            "flying",
        ],
    ),
    group(
        "dynamic_fov",
        "Other changes",
        &["bow_zoom", "underwater", "dying"],
    ),
    // Replay: what a recording shows of your screens.
    group("replay", "Menus and chat", &["show_screens", "chat_text"]),
    // Replay: the export's encoder (a workaround for graphics drivers whose encoder misbehaves).
    group("replay", "Exporting on this PC", &["export_encoder"]),
    // Replay: the sound channels (the microphone's two rows stay under Recording).
    group(
        "replay",
        "Sound channels",
        &[
            "desktop_audio",
            "desktop_source",
            "desktop_volume",
            "game_volume",
            "export_desktop",
            "export_tracks",
        ],
    ),
    // Particles: "Use particle physics"under its own heading, Performance: skipping
    // particle collisions is a performance switch.
    group("particles", "Performance", &["particle_physics"]),
    // Chat: "Symbol shadows" works on all text, with the Chat mod on or off.
    group("chat", "All text", &[crate::mods::chat::SYMBOL_SHADOWS]),
    //  Aero Title Screen: "Check for updates" under its own heading, after Main menu and Logo.
    group("aero_title", "Updates", &["check_for_updates"]),
    group("aero_title", "Accounts", &["accounts"]),
];

/// Paired rows.
pub static PAIRED: &[PairedRows] = &[
    // Chat Macros: one row per macro, the key beside its text (6 rows instead of 12).
    PairedRows {
        mod_id: "chat_macros",
        pairs: &crate::mods::chat_macros::SLOT_IDS,
    },
];

/// Tabbed pickers.
pub static TABBED: &[TabbedPicker] = &[
    // Particles: the group picker becomes tabs above the two sliders it switches.
    TabbedPicker {
        mod_id: "particles",
        picker: "group",
        edits: &["amount", "size"],
    },
];

/// Settings that work whether or not their mod is switched on, as `(mod id, option id)`. The
/// settings window must not tell the player that these "apply when you turn it on".
pub static ALWAYS_APPLY: &[(&str, &str)] = &[
    // Chat history is kept with the Chat mod off.
    ("chat", "keep_chat"),
    ("chat", "history_length"),
    // The replay viewer opens whether or not recording is on.
    ("replay", "open_viewer"),
    // Symbol shadows apply to all text, with the Chat mod on or off.
    ("chat", crate::mods::chat::SYMBOL_SHADOWS),
    // Fog's "Clear water and lava" is read with Fog on or off (its own text
    // says so; `w4_fog_vignette.rs`), and the Accounts button opens the Accounts screen with the
    // Aero title screen off (`w21_accounts.rs`). Without them, the settings window told a player
    // with either mod off that these "apply when you turn it on".
    ("fog", crate::mods::fog::CLEAR_LIQUIDS),
    ("aero_title", "accounts"),
];

/// Whether option `option_id` of `mod_id` works while the mod is off ([`ALWAYS_APPLY`]).
#[must_use]
pub fn applies_while_off(mod_id: &str, option_id: &str) -> bool {
    ALWAYS_APPLY
        .iter()
        .any(|(m, o)| *m == mod_id && *o == option_id)
}

/// The rules about one mod's settings.
pub fn rules_for(mod_id: &str) -> impl Iterator<Item = &'static OptionRule> + '_ {
    RULES.iter().filter(move |rule| rule.mod_id == mod_id)
}

/// The named groups of one mod, in order.
pub fn groups_for(mod_id: &str) -> impl Iterator<Item = &'static OptionGroup> + '_ {
    GROUPS.iter().filter(move |group| group.mod_id == mod_id)
}

// ---- row helpers (any lane may use them in its rows) ---------------------------------------

/// A rule that hides `option` while `when` does not hold (it belongs to another mode). Its hover
/// text, shown when a search finds it anyway, is worked out from `when` ("Only when Mode is TAA").
#[must_use]
pub const fn hide(mod_id: &'static str, option: &'static str, when: Cond) -> OptionRule {
    OptionRule {
        mod_id,
        option,
        when,
        unmet: Unmet::Hide,
    }
}

/// A rule that dims `option`, with `reason` on hover, while `when` does not hold. Write the reason
/// as "Needs <label> on" or "Needs <label> off" with the other setting's label (a test checks it).
#[must_use]
pub const fn dim(
    mod_id: &'static str,
    option: &'static str,
    when: Cond,
    reason: &'static str,
) -> OptionRule {
    OptionRule {
        mod_id,
        option,
        when,
        unmet: Unmet::Dim(reason),
    }
}

/// An open named group.
#[must_use]
pub const fn group(
    mod_id: &'static str,
    title: &'static str,
    options: &'static [&'static str],
) -> OptionGroup {
    OptionGroup {
        mod_id,
        title,
        options,
        collapsed: false,
    }
}

const NEEDS_BACKGROUND: &str = "Needs Background on";
const NEEDS_STATUS_TEXT: &str = "Needs Show status text on";
const NEEDS_SCOREBOARD: &str = "Needs Hide scoreboard off";
const NEEDS_BOSS_BAR: &str = "Needs Hide boss bar off";
const NEEDS_TITLES: &str = "Needs Hide titles off";

// ---- what a setting looks like right now -------------------------------------------------------

/// How the settings window shows one setting right now.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptionState {
    /// It does something: draw it normally.
    Shown,
    /// It does nothing until the rule's condition holds: draw it dimmed (still editable), with
    /// [`OptionRule::reason`] on hover.
    Dimmed(&'static OptionRule),
    /// It belongs to another mode: leave it out (a setting found by search is shown dimmed
    /// instead, see [`OptionState::for_search_hit`]).
    Hidden(&'static OptionRule),
}

impl OptionState {
    /// The state of a setting the player found by search: never hidden, dimmed instead.
    #[must_use]
    pub fn for_search_hit(self) -> OptionState {
        match self {
            OptionState::Hidden(rule) => OptionState::Dimmed(rule),
            other => other,
        }
    }

    /// The rule behind a dimmed or hidden setting.
    #[must_use]
    pub fn rule(self) -> Option<&'static OptionRule> {
        match self {
            OptionState::Shown => None,
            OptionState::Dimmed(rule) | OptionState::Hidden(rule) => Some(rule),
        }
    }
}

impl OptionRule {
    /// The hover text of a dimmed setting, written into `out` (cleared first): the rule's own
    /// reason, or for a hide rule one worked out from its condition and `descriptor`'s labels.
    pub fn reason(&self, descriptor: &ModDescriptor, out: &mut String) {
        out.clear();
        if let Unmet::Dim(reason) = self.unmet {
            out.push_str(reason);
            return;
        }
        let label = |id: &'static str| -> &'static str {
            descriptor.option(id).map_or(id, |option| option.label)
        };
        match self.when {
            Cond::On(id) => {
                out.push_str("Needs ");
                out.push_str(label(id));
                out.push_str(" on");
            }
            Cond::Off(id) => {
                out.push_str("Needs ");
                out.push_str(label(id));
                out.push_str(" off");
            }
            Cond::ChoiceIn(id, indices) => {
                out.push_str("Only when ");
                out.push_str(label(id));
                out.push_str(" is ");
                let choices = match descriptor.option(id).map(|option| option.kind) {
                    Some(OptionKind::Choice { choices }) => choices,
                    _ => &[],
                };
                for (n, index) in indices.iter().enumerate() {
                    if n > 0 {
                        out.push_str(if n + 1 == indices.len() { " or " } else { ", " });
                    }
                    out.push_str(choices.get(*index).copied().unwrap_or("?"));
                }
            }
            Cond::ModOn => out.push_str("Needs the mod on"),
        }
    }
}

/// How the settings window shows option `option_index` of mod `mod_index` (indices into
/// `settings.catalog()` and the mod's `options`) with the current values. Allocation-free: call
/// it per drawn row. A hide rule wins over a dim rule; among dim rules the first in [`RULES`]
/// wins.
#[must_use]
pub fn option_state(settings: &ModSettings, mod_index: usize, option_index: usize) -> OptionState {
    let Some(descriptor) = settings.catalog().get(mod_index) else {
        return OptionState::Shown;
    };
    let Some(option) = descriptor.options.get(option_index) else {
        return OptionState::Shown;
    };
    let mut dimmed = None;
    for rule in rules_for(descriptor.id).filter(|rule| rule.option == option.id) {
        if holds(settings, mod_index, descriptor, rule.when) {
            continue;
        }
        match rule.unmet {
            Unmet::Hide => return OptionState::Hidden(rule),
            Unmet::Dim(_) => {
                dimmed.get_or_insert(rule);
            }
        }
    }
    dimmed.map_or(OptionState::Shown, OptionState::Dimmed)
}

/// [`option_state`] by ids; `Shown` for an unknown mod or option.
#[must_use]
pub fn option_state_by_id(settings: &ModSettings, mod_id: &str, option_id: &str) -> OptionState {
    let Some(mod_index) = settings.index_of(mod_id) else {
        return OptionState::Shown;
    };
    let Some(option_index) = settings.catalog()[mod_index].option_index(option_id) else {
        return OptionState::Shown;
    };
    option_state(settings, mod_index, option_index)
}

/// Whether `cond` holds now. A condition on a setting the mod does not have holds, so a stale
/// row never hides anything (a test keeps the rows honest).
fn holds(settings: &ModSettings, mod_index: usize, descriptor: &ModDescriptor, cond: Cond) -> bool {
    let value = |id: &str| {
        descriptor
            .option_index(id)
            .and_then(|option| settings.value_at(mod_index, option))
    };
    match cond {
        Cond::On(id) => !matches!(value(id), Some(OptionValue::Bool(false))),
        Cond::Off(id) => !matches!(value(id), Some(OptionValue::Bool(true))),
        Cond::ChoiceIn(id, indices) => match value(id) {
            Some(OptionValue::Choice(index)) => indices.contains(index),
            _ => true,
        },
        Cond::ModOn => settings.enabled_at(mod_index),
    }
}

// ---- the settings window's layout ----------------------------------------------------------

/// A heading in the settings window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Heading {
    /// The first block: settings with no heading.
    Top,
    /// A named group from [`GROUPS`].
    Named(&'static str),
    /// Every key setting not in a group or a pair.
    Keys,
    /// Colours, text shadows, backgrounds and brackets not in a group.
    Style,
}

impl Heading {
    /// The heading text (drawn in capitals), `None` for the top block.
    #[must_use]
    pub fn title(self) -> Option<&'static str> {
        match self {
            Heading::Top => None,
            Heading::Named(title) => Some(title),
            Heading::Keys => Some("Keys"),
            Heading::Style => Some("Style"),
        }
    }
}

/// One row of the settings window.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LayoutRow {
    /// One setting, by index into the mod's `options`.
    Option(usize),
    /// Two settings side by side ([`PAIRED`]): `(left, right)`.
    Pair(usize, usize),
    /// A choice drawn as tabs above the settings it switches ([`TABBED`]).
    Tabs {
        /// The choice.
        picker: usize,
        /// The settings drawn under the tabs, which edit the picked entry.
        edits: Vec<usize>,
    },
}

impl LayoutRow {
    /// Every option index this row draws, in order.
    pub fn options(&self) -> impl Iterator<Item = usize> + '_ {
        let (first, second, rest): (usize, Option<usize>, &[usize]) = match self {
            LayoutRow::Option(index) => (*index, None, &[]),
            LayoutRow::Pair(left, right) => (*left, Some(*right), &[]),
            LayoutRow::Tabs { picker, edits } => (*picker, None, edits),
        };
        std::iter::once(first)
            .chain(second)
            .chain(rest.iter().copied())
    }
}

/// A heading and the rows under it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayoutBlock {
    /// The heading.
    pub heading: Heading,
    /// Whether the block starts collapsed (a named group's `collapsed`).
    pub collapsed: bool,
    /// The rows, in order.
    pub rows: Vec<LayoutRow>,
}

/// The settings window of one mod: blocks in drawing order.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SettingsLayout {
    /// The top block (if any), the named groups in table order, then KEYS, then STYLE. Never an
    /// empty block.
    pub blocks: Vec<LayoutBlock>,
}

impl SettingsLayout {
    /// Every option index drawn, in drawing order.
    pub fn options(&self) -> impl Iterator<Item = usize> + '_ {
        self.blocks
            .iter()
            .flat_map(|block| block.rows.iter().flat_map(LayoutRow::options))
    }

    /// The block that draws option `option_index` (jump-to-setting opens a collapsed group).
    #[must_use]
    pub fn block_of(&self, option_index: usize) -> Option<usize> {
        self.blocks.iter().position(|block| {
            block
                .rows
                .iter()
                .any(|row| row.options().any(|index| index == option_index))
        })
    }
}

/// Settings that go under STYLE by their id: the shared HUD style set ([`crate::presets`]) and
/// the backdrop opacities.
const STYLE_IDS: [&str; 5] = [
    crate::presets::TEXT_SHADOW_ID,
    crate::presets::BACKGROUND_ID,
    crate::presets::BACKGROUND_COLOR_ID,
    crate::presets::BRACKETS_ID,
    "background_opacity",
];

/// Whether an ungrouped setting goes under STYLE: a colour, a text shadow, a background or the
/// brackets.
fn is_style(option: &OptionDescriptor) -> bool {
    matches!(option.kind, OptionKind::Color { .. })
        || STYLE_IDS.contains(&option.id)
        || contains_ignore_ascii_case(option.label, "background")
}

fn contains_ignore_ascii_case(hay: &str, needle: &str) -> bool {
    hay.as_bytes()
        .windows(needle.len())
        .any(|window| window.eq_ignore_ascii_case(needle.as_bytes()))
}

/// The settings window of `descriptor`: every visible setting exactly once. Paired rows and
/// tabbed pickers sit in the top block where their first setting is; named groups follow in
/// [`GROUPS`] order; then KEYS (key settings) and STYLE ([`Heading::Style`]). It allocates, so
/// work it out when the selected mod changes, not every frame.
#[must_use]
pub fn settings_layout(descriptor: &ModDescriptor) -> SettingsLayout {
    let options = descriptor.options;
    let mut used: Vec<bool> = options.iter().map(|o| !o.kind.is_visible()).collect();
    // Rows that stand in for several settings, at the index they are drawn at.
    let mut special: Vec<(usize, LayoutRow)> = Vec::new();
    for paired in PAIRED.iter().filter(|p| p.mod_id == descriptor.id) {
        for (left, right) in paired.pairs {
            let (Some(l), Some(r)) = (
                descriptor.option_index(left),
                descriptor.option_index(right),
            ) else {
                continue;
            };
            if l != r && !used[l] && !used[r] {
                used[l] = true;
                used[r] = true;
                special.push((l.min(r), LayoutRow::Pair(l, r)));
            }
        }
    }
    for tabbed in TABBED.iter().filter(|t| t.mod_id == descriptor.id) {
        let Some(picker) = descriptor.option_index(tabbed.picker) else {
            continue;
        };
        if used[picker] {
            continue;
        }
        used[picker] = true;
        let mut edits = Vec::new();
        for id in tabbed.edits {
            if let Some(index) = descriptor.option_index(id).filter(|&i| !used[i]) {
                used[index] = true;
                edits.push(index);
            }
        }
        special.push((picker, LayoutRow::Tabs { picker, edits }));
    }
    special.sort_by_key(|(at, _)| *at);
    let mut named = Vec::new();
    for group in groups_for(descriptor.id) {
        let mut rows = Vec::new();
        for id in group.options {
            if let Some(index) = descriptor.option_index(id).filter(|&i| !used[i]) {
                used[index] = true;
                rows.push(LayoutRow::Option(index));
            }
        }
        if !rows.is_empty() {
            named.push(LayoutBlock {
                heading: Heading::Named(group.title),
                collapsed: group.collapsed,
                rows,
            });
        }
    }
    let (mut top, mut keys, mut style) = (Vec::new(), Vec::new(), Vec::new());
    let mut special = special.into_iter().peekable();
    for (index, option) in options.iter().enumerate() {
        while let Some((_, row)) = special.next_if(|(at, _)| *at == index) {
            top.push(row);
        }
        if used[index] {
            continue;
        }
        if matches!(option.kind, OptionKind::Keybind) {
            keys.push(LayoutRow::Option(index));
        } else if is_style(option) {
            style.push(LayoutRow::Option(index));
        } else {
            top.push(LayoutRow::Option(index));
        }
    }
    let mut blocks = Vec::with_capacity(named.len() + 3);
    let block = |heading, rows| LayoutBlock {
        heading,
        collapsed: false,
        rows,
    };
    if !top.is_empty() {
        blocks.push(block(Heading::Top, top));
    }
    blocks.extend(named);
    if !keys.is_empty() {
        blocks.push(block(Heading::Keys, keys));
    }
    if !style.is_empty() {
        blocks.push(block(Heading::Style, style));
    }
    SettingsLayout { blocks }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog;
    use std::collections::HashSet;

    fn descriptor(id: &str) -> &'static ModDescriptor {
        catalog::find(id).unwrap_or_else(|| panic!("no mod {id}"))
    }

    /// Every row names a real mod and setting; a condition names a setting of the right kind.
    #[test]
    fn every_rule_names_real_settings_of_the_right_kind() {
        for rule in RULES {
            let d = descriptor(rule.mod_id);
            assert!(
                d.option(rule.option).is_some(),
                "{}.{} does not exist",
                rule.mod_id,
                rule.option
            );
            let (cond_id, ok) = match rule.when {
                Cond::On(id) | Cond::Off(id) => (
                    id,
                    matches!(d.option(id).map(|o| o.kind), Some(OptionKind::Toggle)),
                ),
                Cond::ChoiceIn(id, indices) => (
                    id,
                    match d.option(id).map(|o| o.kind) {
                        Some(OptionKind::Choice { choices }) => {
                            !indices.is_empty() && indices.iter().all(|i| *i < choices.len())
                        }
                        _ => false,
                    },
                ),
                Cond::ModOn => ("", true),
            };
            assert!(
                ok,
                "{}.{}: bad condition {:?}",
                rule.mod_id, rule.option, rule.when
            );
            assert_ne!(cond_id, rule.option, "{}: a rule on itself", rule.mod_id);
            // "Needs <label> on/off" names the other setting's label.
            if let (Unmet::Dim(reason), Cond::On(id) | Cond::Off(id)) = (rule.unmet, rule.when) {
                let label = d.option(id).map_or("", |o| o.label);
                let expected_on = format!("Needs {label} on");
                let expected_off = format!("Needs {label} off");
                let want = if matches!(rule.when, Cond::On(_)) {
                    expected_on
                } else {
                    expected_off
                };
                assert!(
                    reason.eq_ignore_ascii_case(&want),
                    "{}.{}: reason {reason:?}, expected {want:?}",
                    rule.mod_id,
                    rule.option
                );
            }
        }
    }

    #[test]
    fn groups_pairs_and_tabs_name_real_settings_once() {
        let mut titles = HashSet::new();
        let mut grouped = HashSet::new();
        for group in GROUPS {
            let d = descriptor(group.mod_id);
            assert!(!group.options.is_empty() && !group.title.is_empty());
            assert!(
                titles.insert((group.mod_id, group.title)),
                "{}: two groups called {}",
                group.mod_id,
                group.title
            );
            for id in group.options {
                let option = d
                    .option(id)
                    .unwrap_or_else(|| panic!("{}.{id} does not exist", group.mod_id));
                assert!(option.kind.is_visible(), "{}.{id} is hidden", group.mod_id);
                assert!(
                    grouped.insert((group.mod_id, *id)),
                    "{}.{id} is in two groups",
                    group.mod_id
                );
            }
        }
        for paired in PAIRED {
            let d = descriptor(paired.mod_id);
            for (left, right) in paired.pairs {
                assert!(d.option(left).is_some() && d.option(right).is_some());
            }
        }
        for tabbed in TABBED {
            let d = descriptor(tabbed.mod_id);
            assert!(matches!(
                d.option(tabbed.picker).map(|o| o.kind),
                Some(OptionKind::Choice { .. })
            ));
            for id in tabbed.edits {
                assert!(d.option(id).is_some(), "{}.{id}", tabbed.mod_id);
            }
        }
        for (mod_id, option_id) in ALWAYS_APPLY {
            assert!(
                descriptor(mod_id).option(option_id).is_some(),
                "ALWAYS_APPLY: {mod_id}.{option_id} does not exist"
            );
        }
        assert!(applies_while_off("chat", "history_length"));
        assert!(!applies_while_off("chat", "stack"));
    }

    /// The settings that work with their mod off are listed, so the settings
    /// window never says they wait for it: Fog's "Clear water and lava" (the client reads it
    /// whatever the switch says) and the Accounts button. The Aero title screen's own settings
    /// (scene, logo, update check) do wait for it.
    #[test]
    fn settings_that_work_with_their_mod_off_are_listed() {
        assert!(applies_while_off("fog", crate::mods::fog::CLEAR_LIQUIDS));
        assert!(applies_while_off("aero_title", "accounts"));
        for waits in ["menu_scene", "logo_style", "check_for_updates"] {
            assert!(!applies_while_off("aero_title", waits), "{waits}");
        }
    }

    /// The layout of every catalog mod draws each visible setting exactly once, and no block is
    /// empty.
    #[test]
    fn every_layout_draws_each_visible_setting_once() {
        for d in catalog::all() {
            let layout = settings_layout(d);
            let drawn: Vec<usize> = layout.options().collect();
            let unique: HashSet<usize> = drawn.iter().copied().collect();
            assert_eq!(drawn.len(), unique.len(), "{}: drawn twice", d.id);
            let visible: HashSet<usize> = d
                .options
                .iter()
                .enumerate()
                .filter(|(_, o)| o.kind.is_visible())
                .map(|(i, _)| i)
                .collect();
            assert_eq!(unique, visible, "{}", d.id);
            for block in &layout.blocks {
                assert!(
                    !block.rows.is_empty(),
                    "{}: empty {:?}",
                    d.id,
                    block.heading
                );
            }
            for index in visible {
                assert!(layout.block_of(index).is_some());
            }
        }
    }

    fn headings(id: &str) -> Vec<Option<&'static str>> {
        settings_layout(descriptor(id))
            .blocks
            .iter()
            .map(|block| block.heading.title())
            .collect()
    }

    #[test]
    fn keys_and_style_get_their_headings_and_groups_their_names() {
        // FPS: "Show" at the top, the five shared style settings under STYLE.
        let fps = settings_layout(descriptor("fps"));
        assert_eq!(fps.blocks.len(), 2);
        assert_eq!(fps.blocks[0].rows, [LayoutRow::Option(0)]);
        assert_eq!(fps.blocks[1].heading, Heading::Style);
        assert_eq!(fps.blocks[1].rows.len(), 5);
        assert_eq!(headings("zoom"), [None, Some("While zoomed"), Some("Keys")]);
        assert_eq!(
            headings("hitboxes"),
            [None, Some("Show for"), Some("Lines")]
        );
        assert_eq!(headings("markers"), [None, Some("Keys"), Some("Style")]);
        let grading = settings_layout(descriptor("color_grading"));
        assert_eq!(grading.blocks[0].rows.len(), 4, "the masters stay on top");
        assert_eq!(grading.blocks[1].heading, Heading::Named("Per channel"));
        assert!(grading.blocks[1].collapsed);
        assert_eq!(grading.blocks[1].rows.len(), 9);
        assert_eq!(
            grading.block_of(4),
            Some(1),
            "slope_r is in the collapsed group"
        );
    }

    /// Chat Macros is six rows of key + text; Particles' picker is tabs above the two
    /// sliders it switches.
    #[test]
    fn chat_macros_pair_up_and_particles_get_tabs() {
        let macros = settings_layout(descriptor("chat_macros"));
        assert_eq!(
            macros.blocks.len(),
            1,
            "no KEYS block: the keys sit beside their text"
        );
        assert_eq!(macros.blocks[0].heading, Heading::Top);
        let rows = &macros.blocks[0].rows;
        assert_eq!(rows.len(), 6);
        assert_eq!(rows[0], LayoutRow::Pair(0, 1));
        assert_eq!(rows[5], LayoutRow::Pair(10, 11));
        let particles = settings_layout(descriptor("particles"));
        let d = descriptor("particles");
        let index = |id| d.option_index(id).unwrap();
        assert_eq!(
            particles.blocks[0].rows[0],
            LayoutRow::Tabs {
                picker: index("group"),
                edits: vec![index("amount"), index("size")],
            }
        );
        assert_eq!(
            particles.blocks[0].rows[1],
            LayoutRow::Option(index("always_crit"))
        );
    }

    #[test]
    fn mode_settings_hide_and_switch_settings_dim() {
        let mut settings = ModSettings::new();
        let state = |s: &ModSettings, m: &str, o: &str| option_state_by_id(s, m, o);
        // Anti-Aliasing starts in TAA: its FXAA setting is hidden, and the other way round.
        assert_eq!(
            state(&settings, "anti_aliasing", "sharpness"),
            OptionState::Shown
        );
        assert!(matches!(
            state(&settings, "anti_aliasing", "texture_edges"),
            OptionState::Hidden(_)
        ));
        settings.set_choice(
            "anti_aliasing",
            "mode",
            crate::mods::anti_aliasing::MODE_FXAA,
        );
        assert!(matches!(
            state(&settings, "anti_aliasing", "sharpness"),
            OptionState::Hidden(_)
        ));
        assert_eq!(
            state(&settings, "anti_aliasing", "texture_edges"),
            OptionState::Shown
        );
        // Crosshair's custom colour shows only with Colour on Custom.
        assert!(matches!(
            state(&settings, "crosshair", "color"),
            OptionState::Hidden(_)
        ));
        settings.set_choice(
            "crosshair",
            "color_mode",
            crate::mods::crosshair::COLOR_MODE_CUSTOM,
        );
        assert_eq!(state(&settings, "crosshair", "color"), OptionState::Shown);
        // A readout's box colour needs its box.
        assert_eq!(
            state(&settings, "fps", "background_color"),
            OptionState::Shown
        );
        settings.set_bool("fps", "background", false);
        let dimmed = state(&settings, "fps", "background_color");
        let mut text = String::new();
        dimmed
            .rule()
            .expect("dimmed")
            .reason(descriptor("fps"), &mut text);
        assert!(matches!(dimmed, OptionState::Dimmed(_)));
        assert_eq!(text, "Needs Background on");
        // Hide switches dim the rest of their mod.
        settings.set_bool("scoreboard", "hide", true);
        assert!(matches!(
            state(&settings, "scoreboard", "hide_numbers"),
            OptionState::Dimmed(_)
        ));
        assert_eq!(state(&settings, "scoreboard", "hide"), OptionState::Shown);
    }

    /// Toggle Sprint's box colour has two rules: the first unmet one gives the reason.
    #[test]
    fn the_first_unmet_rule_explains_a_dimmed_setting() {
        let mut settings = ModSettings::new();
        let d = descriptor("toggle_sprint");
        let mut text = String::new();
        // Fresh install: the status text shows, its box is off.
        let state = option_state_by_id(&settings, "toggle_sprint", "background_color");
        state.rule().expect("dimmed").reason(d, &mut text);
        assert_eq!(text, "Needs Background on");
        settings.set_bool("toggle_sprint", "show_text", false);
        let state = option_state_by_id(&settings, "toggle_sprint", "background_color");
        state.rule().expect("dimmed").reason(d, &mut text);
        assert_eq!(text, "Needs Show status text on");
        settings.set_bool("toggle_sprint", "show_text", true);
        settings.set_bool("toggle_sprint", "background", true);
        assert_eq!(
            option_state_by_id(&settings, "toggle_sprint", "background_color"),
            OptionState::Shown
        );
    }

    /// A hidden setting found by search shows dimmed, with a reason worked out from the mode.
    #[test]
    fn a_search_hit_is_never_hidden_and_says_why() {
        let settings = ModSettings::new();
        let state = option_state_by_id(&settings, "anti_aliasing", "texture_edges");
        assert!(matches!(state, OptionState::Hidden(_)));
        let found = state.for_search_hit();
        assert!(matches!(found, OptionState::Dimmed(_)));
        let mut text = String::new();
        found
            .rule()
            .expect("rule")
            .reason(descriptor("anti_aliasing"), &mut text);
        assert_eq!(text, "Only when Mode is FXAA");
        assert_eq!(OptionState::Shown.for_search_hit(), OptionState::Shown);
        let rule = hide("x", "y", Cond::ChoiceIn("mode", &[0, 1]));
        rule.reason(descriptor("motion_blur"), &mut text);
        assert_eq!(text, "Only when Mode is Frame blend or Camera");
    }

    /// Unknown ids never hide anything.
    #[test]
    fn unknown_mods_and_settings_are_shown() {
        let settings = ModSettings::new();
        assert_eq!(
            option_state_by_id(&settings, "nope", "x"),
            OptionState::Shown
        );
        assert_eq!(
            option_state_by_id(&settings, "fps", "nope"),
            OptionState::Shown
        );
        assert_eq!(option_state(&settings, 9999, 0), OptionState::Shown);
    }
}
