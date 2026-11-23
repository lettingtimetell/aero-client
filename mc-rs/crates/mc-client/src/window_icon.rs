//! The Aero Client logo as the window's icon: the title bar, the taskbar and Alt+Tab.
//!
//! `assets/icon/` holds the logo's source, `aero-client-logo.svg`, and the logo drawn at each size
//! the window can ask for, `aero-<size>.png`: 16 to 32 px for the title bar and 32 to 64 px for the
//! taskbar, at 100 % to 200 % display scaling. Each is drawn at its own size rather than one picture
//! scaled down, so the small ones stay sharp; redraw them all from the SVG after changing it.
//! [`WindowIcon`] picks the sizes for the window's screen.

use mc_render::window_icon::WindowIcon;

/// The built-in pictures: `(side in pixels, PNG bytes)`.
const PICTURES: [(u32, &[u8]); 7] = [
    (16, include_bytes!("../assets/icon/aero-16.png")),
    (20, include_bytes!("../assets/icon/aero-20.png")),
    (24, include_bytes!("../assets/icon/aero-24.png")),
    (32, include_bytes!("../assets/icon/aero-32.png")),
    (40, include_bytes!("../assets/icon/aero-40.png")),
    (48, include_bytes!("../assets/icon/aero-48.png")),
    (64, include_bytes!("../assets/icon/aero-64.png")),
];

/// The logo at every built-in size, for [`mc_render::shell::ShellConfig::icon`]. A picture that
/// doesn't decode to its size is logged and left out; the window then uses the nearest other size.
#[must_use]
pub fn aero_window_icon() -> WindowIcon {
    PICTURES.iter().fold(WindowIcon::default(), |icon, &(side, png)| {
        match image::load_from_memory_with_format(png, image::ImageFormat::Png) {
            Ok(picture) => {
                let rgba = picture.into_rgba8();
                if rgba.width() == side && rgba.height() == side {
                    icon.with_image(side, rgba.into_raw())
                } else {
                    tracing::warn!(side, width = rgba.width(), height = rgba.height(),
                                   "window icon picture has the wrong size; left out");
                    icon
                }
            }
            Err(error) => {
                tracing::warn!(%error, side, "window icon picture didn't decode; left out");
                icon
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_size_is_built_in_and_decodes() {
        assert_eq!(aero_window_icon().sizes(), [16, 20, 24, 32, 40, 48, 64]);
    }

    #[test]
    fn each_picture_is_the_logo_on_a_transparent_background() {
        let icon = aero_window_icon();
        for side in icon.sizes() {
            let (picked, rgba) = icon.pick(side).expect("the size is held");
            assert_eq!(picked, side);
            let alpha = |x: u32, y: u32| rgba[((y * side + x) * 4 + 3) as usize];
            assert_eq!(alpha(0, 0), 0, "{side} px: the top-left corner is transparent");
            assert_eq!(alpha(side - 1, side - 1), 0, "{side} px: the bottom-right corner too");
            assert_eq!(alpha(side / 2, side / 2), 255, "{side} px: the cube's middle is opaque");
        }
    }

    #[test]
    fn the_window_gets_the_sharp_size_for_common_display_scaling() {
        let icon = aero_window_icon();
        for (scale, small, big) in [(1.0, 16, 32), (1.25, 20, 40), (1.5, 24, 48), (2.0, 32, 64)] {
            let (want_small, want_big) = WindowIcon::target_sizes(scale);
            assert_eq!(icon.pick(want_small).map(|(side, _)| side), Some(small), "title bar at {scale}");
            assert_eq!(icon.pick(want_big).map(|(side, _)| side), Some(big), "taskbar at {scale}");
        }
    }
}
