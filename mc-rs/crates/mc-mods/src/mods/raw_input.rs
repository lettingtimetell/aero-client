//! Raw Mouse Input: Reads mouse movement straight from the device, without OS acceleration.
//!
//! Keep the `id` stable (it is the settings-file key); the
//! `category` line must name the mod's home in `crate::menu_layout`, which decides where the
//! menu lists it (a test keeps the two in step).
//!
//! **ON by default**: ON is exactly what mc-rs always did (`WM_INPUT` device counts through winit's
//! `DeviceEvent::MouseMotion`). OFF switches the look to vanilla LWJGL 2's path: OS pointer
//! positions, re-centred after every frame with motion, so Windows' pointer speed and "Enhance
//! pointer precision" apply. The switch reaches the shell through
//! `mc_render::InputState::set_os_pointer_look` (`m3_controls.rs`).

use crate::category::Category;
use crate::descriptor::ModDescriptor;

/// The catalog entry.
pub static DESCRIPTOR: ModDescriptor = ModDescriptor {
    id: "raw_input",
    name: "Raw Mouse Input",
    category: Category::Controls,
    description: "On: mouse look reads the device directly, with no Windows acceleration. Off: \
                  it follows the Windows pointer, acceleration included, like vanilla 1.8.9.",
    default_enabled: true,
    options: &[],
    hud_elements: &[],
};
