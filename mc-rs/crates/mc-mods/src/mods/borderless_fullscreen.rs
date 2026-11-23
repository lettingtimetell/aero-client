//! Borderless Fullscreen: Makes fullscreen a borderless window, for instant alt-tab.
//!
//! Keep the `id` stable (it is the settings-file key); the
//! `category` line must name the mod's home in `crate::menu_layout`, which decides where the
//! menu lists it (a test keeps the two in step).
//!
//! **ON by default**: ON is mc-rs's F11 as it always was, `Fullscreen::Borderless`, the
//! measured Independent Flip fast path. OFF makes F11 an exclusive video-mode switch, as LWJGL 2
//! did; that is opt-in only. The window side is `mc_render::shell::apply_fullscreen`, reached
//! through `ShellApp::wants_exclusive_fullscreen`.

use crate::category::Category;
use crate::descriptor::ModDescriptor;

/// The catalog entry.
pub static DESCRIPTOR: ModDescriptor = ModDescriptor {
    id: "borderless_fullscreen",
    name: "Borderless Fullscreen",
    category: Category::Controls,
    description: "On: F11 fullscreen is a borderless window, with instant alt-tab and the \
                  fastest display path. Off: F11 switches the monitor's video mode (exclusive).",
    default_enabled: true,
    options: &[],
    hud_elements: &[],
};
