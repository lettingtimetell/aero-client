//! Background FPS: a low frame cap while the game window is in the background.
//!
//! Keep the `id` and the option ids stable (they are settings-file keys); the `category` line
//! must name the mod's home in `crate::menu_layout`, which decides where the menu lists it (a test
//! keeps the two in step).
//!
//! **ON by default**: ON changes nothing while the game has the focus, so the
//! game plays exactly as before; it only spares the CPU and the GPU while another window has the
//! focus or the game is minimized. OFF is vanilla: the same frame rate in the background as in
//! front. The client side is `mc_client::app::mods::background_fps` (the cap, and what never gets
//! one: video exports, the loading screen, diagnostic runs); the shell waits for a background frame
//! in its event loop, so switching back starts the next frame at once
//! (`mc_render::ShellApp::background_frame_cap`).
//!
//! **Both defaults are 30** (frames 33 ms apart): the game runs 20 ticks a second, so a frame runs
//! at most one tick and the player's movement packets keep vanilla's one-per-tick spacing, as at
//! any normal frame rate. Below 20 frames a second a frame runs two or more ticks back to back and
//! those packets leave in bursts, as vanilla's do on a slow PC; the sliders allow it (down to 5,
//! the CPU saving is a little larger) and their descriptions say so. The game clock catches up at
//! most ten ticks a frame (`Timer.elapsedTicks`), so every stop keeps all 20 ticks a second.

use crate::category::Category;
use crate::descriptor::{ModDescriptor, OptionDescriptor, OptionKind, OptionValue};

/// The mod's id.
pub const ID: &str = "background_fps";
/// The frame cap while another window has the focus and the game is still on screen.
pub const UNFOCUSED_FPS: &str = "unfocused_fps";
/// The frame cap while the game window is minimized.
pub const MINIMIZED_FPS: &str = "minimized_fps";

/// The catalog entry.
pub static DESCRIPTOR: ModDescriptor = ModDescriptor {
    id: ID,
    name: "Background FPS",
    category: Category::Controls,
    description: "On: while another window has the focus or the game is minimized, the game \
                  draws fewer frames, to save CPU and GPU power. Full speed comes back the moment \
                  you switch back. Off: the same frame rate in the background as in front.",
    default_enabled: true,
    options: &[
        OptionDescriptor {
            id: UNFOCUSED_FPS,
            label: "Unfocused FPS",
            kind: OptionKind::Slider {
                min: 5.0,
                max: 240.0,
                step: 5.0,
                unit: "",
                decimals: 0,
            },
            default: OptionValue::Float(30.0),
            description: "Frames per second while another window has the focus and the game is \
                          still on screen. Never more than your Max Framerate. Below 20, the \
                          game sends its movement updates in bursts, like a slow PC.",
        },
        OptionDescriptor {
            id: MINIMIZED_FPS,
            label: "Minimized FPS",
            kind: OptionKind::Slider {
                min: 5.0,
                max: 60.0,
                step: 5.0,
                unit: "",
                decimals: 0,
            },
            default: OptionValue::Float(30.0),
            description: "Frames per second while the game is minimized. The game itself keeps \
                          running at full speed. Below 20, it sends its movement updates in \
                          bursts, like a slow PC.",
        },
    ],
    hud_elements: &[],
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ModSettings;

    /// A fresh install: on, 30 unfocused and 30 minimized, read back through the store. At 30
    /// frames a second (33 ms apart) a frame never runs more than one of the 20 ticks a second.
    #[test]
    fn defaults_are_on_and_keep_one_tick_per_frame_at_most() {
        let settings = ModSettings::new();
        assert!(settings.enabled(ID));
        for option in [UNFOCUSED_FPS, MINIMIZED_FPS] {
            let fps = settings.f32(ID, option);
            assert_eq!(fps, 30.0, "{option}");
            assert!(
                1000.0 / fps < 50.0,
                "{option}: frames closer than one tick apart"
            );
        }
    }

    /// The settings file keeps both values and the switch, and a file without the mod (written
    /// before it existed) loads with the defaults; values outside a slider's range are clamped.
    #[test]
    fn the_file_round_trips_and_old_or_odd_files_load_sanely() {
        let mut settings = ModSettings::new();
        settings.set_enabled(ID, false);
        settings.set_f32(ID, UNFOCUSED_FPS, 60.0);
        settings.set_f32(ID, MINIMIZED_FPS, 5.0);
        let text = settings.to_json_string();
        let (loaded, _) = ModSettings::from_json_str(&text);
        assert!(!loaded.enabled(ID));
        assert_eq!(loaded.f32(ID, UNFOCUSED_FPS), 60.0);
        assert_eq!(loaded.f32(ID, MINIMIZED_FPS), 5.0);

        let (old, _) = ModSettings::from_json_str(r#"{"mods":{"fps":{"enabled":true}}}"#);
        assert!(old.enabled(ID), "a file from before the mod: on");
        assert_eq!(old.f32(ID, UNFOCUSED_FPS), 30.0);
        assert_eq!(old.f32(ID, MINIMIZED_FPS), 30.0);

        let odd = r#"{"mods":{"background_fps":{"enabled":true,
            "options":{"unfocused_fps":1000.0,"minimized_fps":0.0}}}}"#;
        let (odd, _) = ModSettings::from_json_str(odd);
        assert_eq!(odd.f32(ID, UNFOCUSED_FPS), 240.0, "clamped to the top stop");
        assert_eq!(odd.f32(ID, MINIMIZED_FPS), 5.0, "never below 5");
    }
}
