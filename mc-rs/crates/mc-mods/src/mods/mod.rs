//! One file per mod. Each holds that mod's `DESCRIPTOR`; the lane that implements the mod
//! owns the file. [`crate::catalog::all`] lists them in menu order.

// The Aero title screen.
pub mod aero_title;
pub mod armor_status;
pub mod auto_gg;
pub mod block_overlay;
pub mod borderless_fullscreen;
pub mod boss_bar;
pub mod centered_inventory;
pub mod chat;
pub mod color_grading;
pub mod connected_textures;
pub mod coordinates;
pub mod cps;
pub mod crosshair;
pub mod custom_sky;
pub mod dynamic_fov;
pub mod fire_height;
pub mod fog;
pub mod fps;
pub mod fullbright;
pub mod independent_gui;
pub mod item_info;
pub mod keystrokes;
pub mod motion_blur;
pub mod no_hurt_cam;
pub mod old_animations;
pub mod particles;
pub mod ping;
pub mod potion_status;
pub mod raw_input;
pub mod reach_display;
pub mod scoreboard;
pub mod shiny_pots;
pub mod spotify;
pub mod tall_grass;
pub mod time_changer;
pub mod toggle_sprint;
pub mod vignette;
pub mod waypoints;
pub mod zoom;
// One file per mods.
pub mod action_bar;
pub mod ambient_occlusion;
pub mod anisotropic_filtering;
pub mod anti_aliasing;
pub mod cape_physics;
pub mod chat_macros;
pub mod hitboxes;
pub mod markers;
pub mod menu_blur;
pub mod modern_keybinds;
pub mod nametags;
pub mod scrollable_tooltips;
pub mod skins_3d;
pub mod sky;
pub mod stuck_arrows;
pub mod super_secret_settings;
pub mod tab_list;
pub mod titles;
pub mod volumetric_clouds;
// One file per mod.
pub mod replay;
pub mod shaders_classic;
pub mod shaders_dreamy;
pub mod shaders_underwater;
// The pause menu's Disconnect asks first.
pub mod smart_disconnect;
// A low frame cap while the game window is in the background.
pub mod background_fps;
