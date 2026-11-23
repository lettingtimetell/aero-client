//! The window's icon: one picture at several sizes, from which the shell picks, for each place the
//! system shows it, the size that suits the window's display scaling.
//!
//! On Windows the title bar uses the small icon (16 logical pixels) and the taskbar and Alt+Tab the
//! big one (32 logical pixels). Handing each a picture drawn at its exact pixel size keeps it sharp:
//! Windows would otherwise scale one picture itself. Elsewhere winit takes one picture, the big one.

use winit::window::{Icon, Window};

/// The logical sizes of the title bar's icon and of the taskbar's (and Alt+Tab's).
const SMALL_LOGICAL: f64 = 16.0;
const BIG_LOGICAL: f64 = 32.0;

/// One icon at several square sizes, each straight (not premultiplied) RGBA8, row by row.
#[derive(Clone, Default)]
pub struct WindowIcon {
    /// `(side in pixels, side * side * 4 bytes)`, smallest first.
    images: Vec<(u32, Vec<u8>)>,
}

impl std::fmt::Debug for WindowIcon {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The sizes, not thousands of pixel bytes.
        f.debug_struct("WindowIcon").field("sizes", &self.sizes()).finish()
    }
}

impl WindowIcon {
    /// Adds a `side` x `side` picture (replacing one of the same size).
    ///
    /// # Panics
    /// When `rgba` is not `side * side * 4` bytes: the pictures are built in, so a wrong length is
    /// a programming error, not a runtime condition.
    #[must_use]
    pub fn with_image(mut self, side: u32, rgba: Vec<u8>) -> Self {
        let side_px = side as usize;
        assert_eq!(rgba.len(), side_px * side_px * 4, "a {side} px icon needs {side}x{side}x4 bytes");
        self.images.retain(|(s, _)| *s != side);
        self.images.push((side, rgba));
        self.images.sort_by_key(|(s, _)| *s);
        self
    }

    /// The sizes held, smallest first.
    #[must_use]
    pub fn sizes(&self) -> Vec<u32> {
        self.images.iter().map(|(side, _)| *side).collect()
    }

    /// Whether no picture has been added.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.images.is_empty()
    }

    /// The smallest picture at least `px` pixels wide, or the largest one when none is that big.
    #[must_use]
    pub fn pick(&self, px: u32) -> Option<(u32, &[u8])> {
        self.images
            .iter()
            .find(|(side, _)| *side >= px)
            .or_else(|| self.images.last())
            .map(|(side, rgba)| (*side, rgba.as_slice()))
    }

    /// The pixel sizes the title bar's icon and the taskbar's want at `scale_factor`.
    #[must_use]
    pub fn target_sizes(scale_factor: f64) -> (u32, u32) {
        // Scale factors are small positive numbers (1.0 to about 4.0), so the casts can't overflow.
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let px = |logical: f64| (logical * scale_factor).round().max(1.0) as u32;
        (px(SMALL_LOGICAL), px(BIG_LOGICAL))
    }

    /// Sets this icon on `window`, at the sizes its current scale factor wants. A picture winit
    /// refuses is logged and left out (the system's default icon shows instead).
    pub fn apply(&self, window: &Window) {
        let (small, big) = Self::target_sizes(window.scale_factor());
        let icon = |px: u32| {
            let (side, rgba) = self.pick(px)?;
            Icon::from_rgba(rgba.to_vec(), side, side)
                .map_err(|error| tracing::warn!(%error, side, "window icon refused"))
                .ok()
        };
        #[cfg(windows)]
        {
            use winit::platform::windows::WindowExtWindows;
            window.set_window_icon(icon(small));
            window.set_taskbar_icon(icon(big));
        }
        #[cfg(not(windows))]
        {
            let _ = small;
            window.set_window_icon(icon(big));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(side: u32) -> Vec<u8> {
        vec![255; (side * side * 4) as usize]
    }

    fn icon(sizes: &[u32]) -> WindowIcon {
        sizes.iter().fold(WindowIcon::default(), |icon, &side| icon.with_image(side, solid(side)))
    }

    #[test]
    fn picks_the_smallest_picture_at_least_as_big_as_asked() {
        let icon = icon(&[64, 16, 32, 24]);
        assert_eq!(icon.sizes(), [16, 24, 32, 64], "kept smallest first, whatever the order added");
        assert_eq!(icon.pick(16).map(|(side, _)| side), Some(16), "an exact size");
        assert_eq!(icon.pick(20).map(|(side, _)| side), Some(24), "the next size up");
        assert_eq!(icon.pick(40).map(|(side, _)| side), Some(64));
        assert_eq!(icon.pick(128).map(|(side, _)| side), Some(64), "none big enough: the largest");
        assert_eq!(icon.pick(1).map(|(side, _)| side), Some(16));
        assert!(WindowIcon::default().pick(16).is_none());
        assert!(WindowIcon::default().is_empty());
    }

    #[test]
    fn the_sizes_follow_display_scaling() {
        assert_eq!(WindowIcon::target_sizes(1.0), (16, 32));
        assert_eq!(WindowIcon::target_sizes(1.25), (20, 40));
        assert_eq!(WindowIcon::target_sizes(1.5), (24, 48));
        assert_eq!(WindowIcon::target_sizes(1.75), (28, 56));
        assert_eq!(WindowIcon::target_sizes(2.0), (32, 64));
        assert_eq!(WindowIcon::target_sizes(0.01), (1, 1), "never zero pixels");
    }

    #[test]
    fn a_size_added_again_replaces_the_first_and_debug_shows_only_sizes() {
        let icon = icon(&[32]).with_image(32, vec![0; 32 * 32 * 4]);
        assert_eq!(icon.sizes(), [32]);
        assert_eq!(icon.pick(32).map(|(_, rgba)| rgba[0]), Some(0), "the second picture won");
        assert_eq!(format!("{icon:?}"), "WindowIcon { sizes: [32] }");
    }

    #[test]
    #[should_panic(expected = "a 16 px icon needs 16x16x4 bytes")]
    fn a_picture_of_the_wrong_length_is_a_programming_error() {
        let _ = WindowIcon::default().with_image(16, vec![0; 10]);
    }
}
