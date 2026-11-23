//! GUI scaling and window↔GUI coordinate mapping.
//!
//! java: `client/gui/ScaledResolution.java`, `client/gui/GuiScreen.java:604`
//! (`handleMouseInput`)
//!
//! Every screen lays itself out in *scaled* pixels, a virtual framebuffer of at least 320×240
//! that the window is an integer multiple of. The scale factor is picked so that the scaled
//! surface never drops below 320×240, capped by the `guiScale` option.

use mc_render::GameSettings;

/// Vanilla's minimum scaled width. Below this the scale factor stops growing.
pub const MIN_SCALED_WIDTH: i32 = 320;
/// Vanilla's minimum scaled height.
pub const MIN_SCALED_HEIGHT: i32 = 240;
/// `guiScale = 0` means "auto"; vanilla implements it as an effectively unbounded cap.
pub const AUTO_SCALE_CAP: i32 = 1000;

/// Whether the GUI projection is vanilla's exact one. On unless `MC_NW_TEXT_ORTHO=0`; read once per process.
#[must_use]
pub fn exact_ortho() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("MC_NW_TEXT_ORTHO").is_none_or(|v| v != "0"))
}

/// Whether Minecraft-font glyphs use the sharp sampler (see `widget::gpu`, `fs_text`). On
/// unless `MC_NW_TEXT_SHARP=0`; read once per process.
#[must_use]
pub fn sharp_text() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("MC_NW_TEXT_SHARP").is_none_or(|v| v != "0"))
}

/// The scaled GUI surface for one window size.
///
/// Cheap to construct — recompute it on every resize rather than caching one, exactly as
/// vanilla does.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScaledResolution {
    display_width: i32,
    display_height: i32,
    scaled_width: i32,
    scaled_height: i32,
    scaled_width_d: f64,
    scaled_height_d: f64,
    scale_factor: i32,
}

impl ScaledResolution {
    /// Compute the scaled surface.
    ///
    /// `gui_scale` is `options.txt`'s `guiScale`: 0 auto, 1 small, 2 normal, 3 large. `unicode`
    /// is the font renderer's unicode flag — vanilla drops an **odd** scale factor by one when
    /// the unicode font is in use, because those glyphs are drawn at half scale and an odd
    /// factor makes them land on half-pixels.
    // java: ScaledResolution.java:14
    #[must_use]
    pub fn new(display_width: u32, display_height: u32, gui_scale: i32, unicode: bool) -> Self {
        let dw = display_width as i32;
        let dh = display_height as i32;
        let mut scale_factor = 1;
        let cap = if gui_scale == 0 {
            AUTO_SCALE_CAP
        } else {
            gui_scale
        };

        while scale_factor < cap
            && dw / (scale_factor + 1) >= MIN_SCALED_WIDTH
            && dh / (scale_factor + 1) >= MIN_SCALED_HEIGHT
        {
            scale_factor += 1;
        }

        if unicode && scale_factor % 2 != 0 && scale_factor != 1 {
            scale_factor -= 1;
        }

        let scaled_width_d = f64::from(dw) / f64::from(scale_factor);
        let scaled_height_d = f64::from(dh) / f64::from(scale_factor);

        Self {
            display_width: dw,
            display_height: dh,
            scaled_width: mc_math::math_helper::ceil_f64(scaled_width_d),
            scaled_height: mc_math::math_helper::ceil_f64(scaled_height_d),
            scaled_width_d,
            scaled_height_d,
            scale_factor,
        }
    }

    /// [`ScaledResolution::new`] reading `guiScale` from parsed `options.txt`.
    ///
    /// `unicode` still has to come from the caller, because it is the *or* of
    /// `forceUnicodeFont` and the active language's own heuristic
    /// ([`mc_assets::Language::is_unicode`]) — which `GameSettings` alone cannot know.
    #[must_use]
    pub fn from_settings(
        display_width: u32,
        display_height: u32,
        settings: &GameSettings,
        unicode: bool,
    ) -> Self {
        Self::new(
            display_width,
            display_height,
            settings.gui_scale,
            unicode || settings.force_unicode_font,
        )
    }

    /// Scaled width in GUI pixels, rounded **up**.
    #[must_use]
    pub const fn scaled_width(&self) -> i32 {
        self.scaled_width
    }

    /// Scaled height in GUI pixels, rounded **up**.
    #[must_use]
    pub const fn scaled_height(&self) -> i32 {
        self.scaled_height
    }

    /// Exact scaled width, before rounding. Vanilla uses this for the projection matrix.
    #[must_use]
    pub const fn scaled_width_f64(&self) -> f64 {
        self.scaled_width_d
    }

    /// Exact scaled height, before rounding.
    #[must_use]
    pub const fn scaled_height_f64(&self) -> f64 {
        self.scaled_height_d
    }

    /// The integer scale factor, 1 or greater.
    #[must_use]
    pub const fn scale_factor(&self) -> i32 {
        self.scale_factor
    }

    /// The window size this was computed from.
    #[must_use]
    pub const fn display_size(&self) -> (i32, i32) {
        (self.display_width, self.display_height)
    }

    /// Map an LWJGL mouse position (origin **bottom**-left) to GUI pixels.
    ///
    /// This is the arithmetic vanilla actually runs, kept so that a parity trace against the
    /// Java client compares like with like. Callers driving winit want
    /// [`cursor_to_gui`](Self::cursor_to_gui) instead.
    // java: GuiScreen.java:606
    #[must_use]
    pub fn lwjgl_to_gui(&self, event_x: i32, event_y: i32) -> (i32, i32) {
        if self.display_width <= 0 || self.display_height <= 0 {
            return (0, 0);
        }
        let x = event_x * self.scaled_width / self.display_width;
        let y = self.scaled_height - event_y * self.scaled_height / self.display_height - 1;
        (x, y)
    }

    /// Map a window cursor position (origin **top**-left, as winit reports) to GUI pixels.
    ///
    /// Equivalent to flipping into LWJGL's frame and calling
    /// [`lwjgl_to_gui`](Self::lwjgl_to_gui), so the off-by-one at the bottom edge matches
    /// vanilla exactly.
    #[must_use]
    pub fn cursor_to_gui(&self, x: f64, y: f64) -> (i32, i32) {
        if self.display_width <= 0 || self.display_height <= 0 {
            return (0, 0);
        }
        let px = (x as i32).clamp(0, self.display_width - 1);
        let py = (y as i32).clamp(0, self.display_height - 1);
        self.lwjgl_to_gui(px, self.display_height - 1 - py)
    }

    /// The x of [`cursor_to_gui`](Self::cursor_to_gui) before its integer division: the same
    /// physical pixel times `scaled_width / display_width` as an exact fraction, so it floors
    /// to `cursor_to_gui(..).0`. Option sliders use it to resolve physical pixels (X1, item 33; see `widget::set_precise_cursor_x`).
    #[must_use]
    pub fn cursor_to_gui_precise_x(&self, x: f64) -> Option<f32> {
        if self.display_width <= 0 || self.display_height <= 0 {
            return None;
        }
        let px = (x as i32).clamp(0, self.display_width - 1);
        let exact = f64::from(px) * f64::from(self.scaled_width) / f64::from(self.display_width);
        Some(exact as f32)
    }

    /// Map this surface's GUI coordinates into another GUI surface on the same window.
    ///
    /// Under vanilla's projection ([`Self::ortho`]) one GUI pixel is exactly `scale_factor`
    /// physical pixels on every surface, so the map between two surfaces of one window is the
    /// ratio of their factors: a surface at factor 2 drawn into one at factor 3 lands on exactly
    /// the pixels a vanilla client at GUI scale 2 would use. (While the projection used the
    /// rounded sizes, the rounded-extent ratio was the matching map; `MC_NW_TEXT_ORTHO=0`
    /// restores both.)
    #[must_use]
    pub fn transform_to(&self, destination: Self) -> super::Transform {
        if !exact_ortho() {
            return super::Transform::scale_xy(
                destination.scaled_width.max(1) as f32 / self.scaled_width.max(1) as f32,
                destination.scaled_height.max(1) as f32 / self.scaled_height.max(1) as f32,
            );
        }
        let s = self.scale_factor.max(1) as f32 / destination.scale_factor.max(1) as f32;
        super::Transform::scale_xy(s, s)
    }

    /// A column-major orthographic projection mapping GUI pixels to clip space.
    ///
    /// `(0, 0)` is the top-left corner and `(scaled_width_f64, scaled_height_f64)` the
    /// bottom-right, with `z = 0`. Written out by hand rather than via `glam` so the crate keeps
    /// its dependency list to what it actually needs.
    ///
    /// This uses the **exact** scaled size, as vanilla's `setupOverlayRendering` does
    /// (`ortho(0, getScaledWidth_double(), getScaledHeight_double(), 0, ...)`), so one GUI pixel
    /// is exactly `scale_factor` physical pixels. This used
    /// the rounded-up integer size, which on a window that is not a multiple of the factor
    /// (1366x768 and 1280x720 at 3, 2560x1440 at 3 or 6, most maximised windows' heights) drew
    /// the GUI at a non-integer scale such as 2.9956, so every few hundred GUI pixels a row or
    /// column of every glyph and icon was dropped: the "squashed text". `MC_NW_TEXT_ORTHO=0`
    /// (read once) restores the old matrix for comparisons.
    // java: EntityRenderer.java:1754 (setupOverlayRendering)
    #[must_use]
    pub fn ortho(&self) -> [[f32; 4]; 4] {
        // A window is at least one pixel wide whenever it draws, which makes the exact size at
        // least 1 (the factor is 1 below 640x480); `max` only guards a zero-area window.
        let (w, h) = if exact_ortho() {
            (
                self.scaled_width_d.max(1.0) as f32,
                self.scaled_height_d.max(1.0) as f32,
            )
        } else {
            (
                self.scaled_width.max(1) as f32,
                self.scaled_height.max(1) as f32,
            )
        };
        [
            [2.0 / w, 0.0, 0.0, 0.0],
            [0.0, -2.0 / h, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0, 1.0],
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The scale factor at a matrix of real window sizes and every `guiScale` setting.
    ///
    #[test]
    fn independent_surface_transform_matches_projection_and_cursor_on_odd_windows() {
        // The size each surface's projection spans: exact under vanilla's matrix, rounded
        // under the `MC_NW_TEXT_ORTHO=0` comparison switch.
        let span = |r: &ScaledResolution| {
            if exact_ortho() {
                (r.scaled_width_f64() as f32, r.scaled_height_f64() as f32)
            } else {
                (r.scaled_width() as f32, r.scaled_height() as f32)
            }
        };
        for (w, h) in [(1920, 1080), (1281, 721), (1921, 1081), (640, 360)] {
            for unicode in [false, true] {
                let global = ScaledResolution::new(w, h, 3, unicode);
                for scale in [0, 1, 2, 3] {
                    let local = ScaledResolution::new(w, h, scale, unicode);
                    let transform = local.transform_to(global);
                    let (lw, lh) = span(&local);
                    let (gw, gh) = span(&global);
                    for (x, y) in [(0.0, 0.0), (411.0, 315.0), (w as f64 - 1.0, h as f64 - 1.0)] {
                        let (gx, gy) = local.cursor_to_gui(x, y);
                        let mapped = transform.apply([gx as f32, gy as f32]);
                        // Both projections put the point on the same physical location.
                        for (actual, expected) in [
                            (mapped[0] / gw, gx as f32 / lw),
                            (mapped[1] / gh, gy as f32 / lh),
                        ] {
                            assert!((actual - expected).abs() < 1e-6);
                        }
                    }
                    let edge = transform.apply([lw, lh]);
                    assert!((edge[0] - gw).abs() < 0.001);
                    assert!((edge[1] - gh).abs() < 0.001);
                }
            }
        }
    }

    /// One GUI pixel is exactly `scale_factor` physical
    /// pixels on every window, including the ones that are not a multiple of the factor, as
    /// vanilla's `ortho(0, getScaledWidth_double(), ...)` makes it. The rounded matrix this
    /// replaced fails here at 1366x768 (455.33 -> 456 columns), 1280x720 and 2560x1440 at 3.
    #[test]
    fn the_projection_is_an_exact_integer_scale_on_every_window() {
        if !exact_ortho() {
            eprintln!("SKIP: MC_NW_TEXT_ORTHO=0");
            return;
        }
        for (w, h) in [
            (1280, 720),
            (1366, 768),
            (1600, 900),
            (1920, 1080),
            (2560, 1440),
            (1920, 1017),
            (2560, 1377),
            (1281, 721),
        ] {
            for gui in 0..=4 {
                let r = ScaledResolution::new(w, h, gui, false);
                let m = r.ortho();
                let f = r.scale_factor() as f64;
                // GUI x -> physical x: (ndc + 1) / 2 * width.
                let to_px = |x: f32, y: f32| {
                    let nx = f64::from(m[0][0] * x + m[3][0]);
                    let ny = f64::from(m[1][1] * y + m[3][1]);
                    (
                        (nx + 1.0) / 2.0 * f64::from(w),
                        (1.0 - ny) / 2.0 * f64::from(h),
                    )
                };
                for g in [0.0f32, 1.0, 17.0, 100.0, 333.0] {
                    let (px, py) = to_px(g, g);
                    assert!(
                        (px - f64::from(g) * f).abs() < 2e-3
                            && (py - f64::from(g) * f).abs() < 2e-3,
                        "{w}x{h} gui {gui}: GUI {g} -> ({px}, {py}), want {}",
                        f64::from(g) * f
                    );
                }
            }
        }
    }

    #[test]
    fn the_surface_transform_is_the_factor_ratio() {
        if !exact_ortho() {
            eprintln!("SKIP: MC_NW_TEXT_ORTHO=0");
            return;
        }
        let global = ScaledResolution::new(1366, 768, 3, false);
        let local = ScaledResolution::new(1366, 768, 2, false);
        let [x, y] = local.transform_to(global).apply([10.0, 30.0]);
        // Local GUI 10 is physical 20 is global 6.67: the same pixel through both projections.
        assert!(
            (x - 20.0 / 3.0).abs() < 1e-5 && (y - 20.0).abs() < 1e-5,
            "({x}, {y})"
        );
    }

    /// Rows are `(width, height, gui_scale, unicode) -> (scale_factor, scaled_w, scaled_h)`,
    /// worked out from `ScaledResolution.java` by hand.
    #[test]
    fn scale_factor_matrix() {
        let cases: &[(u32, u32, i32, bool, i32, i32, i32)] = &[
            // vanilla's default window
            (854, 480, 0, false, 2, 427, 240),
            (854, 480, 1, false, 1, 854, 480),
            (854, 480, 2, false, 2, 427, 240),
            // 3 is capped by the 320x240 floor: 854/3 = 284 < 320
            (854, 480, 3, false, 2, 427, 240),
            // 720p
            (1280, 720, 0, false, 3, 427, 240),
            (1280, 720, 2, false, 2, 640, 360),
            (1280, 720, 3, false, 3, 427, 240),
            // 1080p — 1080/5 = 216 < 240 stops the climb at 4
            (1920, 1080, 0, false, 4, 480, 270),
            (1920, 1080, 1, false, 1, 1920, 1080),
            (1920, 1080, 3, false, 3, 640, 360),
            // 4K
            (3840, 2160, 0, false, 9, 427, 240),
            (3840, 2160, 3, false, 3, 1280, 720),
            // below the floor: the factor never leaves 1
            (320, 240, 0, false, 1, 320, 240),
            (319, 239, 0, false, 1, 319, 239),
            (1, 1, 0, false, 1, 1, 1),
            // exactly twice the floor
            (640, 480, 0, false, 2, 320, 240),
            (639, 480, 0, false, 1, 639, 480),
        ];
        for &(w, h, scale, uni, sf, sw, sh) in cases {
            let r = ScaledResolution::new(w, h, scale, uni);
            assert_eq!(
                (r.scale_factor(), r.scaled_width(), r.scaled_height()),
                (sf, sw, sh),
                "{w}x{h} guiScale={scale} unicode={uni}"
            );
        }
    }

    #[test]
    fn the_unicode_font_forces_an_even_scale_factor() {
        // 4K auto lands on 9, which is odd, so unicode drops it to 8.
        let plain = ScaledResolution::new(3840, 2160, 0, false);
        let uni = ScaledResolution::new(3840, 2160, 0, true);
        assert_eq!(plain.scale_factor(), 9);
        assert_eq!(uni.scale_factor(), 8);
        assert_eq!((uni.scaled_width(), uni.scaled_height()), (480, 270));

        // 1 is exempt: it is odd but must not become 0.
        let small = ScaledResolution::new(400, 300, 0, true);
        assert_eq!(small.scale_factor(), 1);

        // guiScale=3 on 1080p is odd too.
        assert_eq!(ScaledResolution::new(1920, 1080, 3, true).scale_factor(), 2);
        // An even factor is left alone.
        assert_eq!(ScaledResolution::new(1920, 1080, 0, true).scale_factor(), 4);
    }

    #[test]
    fn the_scaled_size_rounds_up_and_the_float_size_does_not() {
        // 1000 / 3 = 333.33
        let r = ScaledResolution::new(1000, 800, 3, false);
        assert_eq!(r.scale_factor(), 3);
        assert_eq!(r.scaled_width(), 334);
        assert_eq!(r.scaled_height(), 267);
        assert!((r.scaled_width_f64() - 1000.0 / 3.0).abs() < 1e-12);
        assert!((r.scaled_height_f64() - 800.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn mouse_mapping_is_vanillas_integer_arithmetic() {
        let r = ScaledResolution::new(1000, 1000, 2, false);
        assert_eq!((r.scaled_width(), r.scaled_height()), (500, 500));
        // LWJGL: y grows upward, so the top of the screen is y = displayHeight - 1.
        assert_eq!(r.lwjgl_to_gui(0, 999), (0, 0));
        assert_eq!(r.lwjgl_to_gui(0, 0), (0, 499), "the bottom row");
        assert_eq!(r.lwjgl_to_gui(999, 999), (499, 0));
        // winit: y grows downward.
        assert_eq!(r.cursor_to_gui(0.0, 0.0), (0, 0));
        assert_eq!(r.cursor_to_gui(999.0, 999.0), (499, 499));
        assert_eq!(r.cursor_to_gui(500.0, 500.0), (250, 250));
    }

    #[test]
    fn a_cursor_outside_the_window_is_clamped_not_wrapped() {
        let r = ScaledResolution::new(800, 600, 2, false);
        assert_eq!(r.cursor_to_gui(-50.0, -50.0), (0, 0));
        assert_eq!(r.cursor_to_gui(9999.0, 9999.0), (399, 299));
    }

    #[test]
    fn non_divisible_height_keeps_vanillas_bottom_left_rounding() {
        let r = ScaledResolution::new(1280, 601, 2, false);
        assert_eq!((r.scaled_width(), r.scaled_height()), (640, 301));
        assert_eq!(r.cursor_to_gui(0.0, 1.0), (0, 1));

        // A direct top-left ratio rounds this to zero and is therefore not equivalent to
        // GuiScreen's bottom-left integer conversion plus final `- 1`.
        let direct_top_left = (1.0 * f64::from(r.scaled_height()) / 601.0) as i32;
        assert_eq!(direct_top_left, 0);
    }

    #[test]
    fn a_zero_sized_window_does_not_divide_by_zero() {
        let r = ScaledResolution::new(0, 0, 0, false);
        assert_eq!(r.scale_factor(), 1);
        assert_eq!(r.cursor_to_gui(10.0, 10.0), (0, 0));
        assert_eq!(r.lwjgl_to_gui(10, 10), (0, 0));
    }

    #[test]
    fn ortho_maps_the_corners_to_clip_space() {
        let r = ScaledResolution::new(1920, 1080, 0, false);
        let m = r.ortho();
        let apply = |x: f32, y: f32| -> (f32, f32) {
            (
                m[0][0] * x + m[1][0] * y + m[3][0],
                m[0][1] * x + m[1][1] * y + m[3][1],
            )
        };
        assert_eq!(apply(0.0, 0.0), (-1.0, 1.0), "top-left");
        assert_eq!(apply(480.0, 270.0), (1.0, -1.0), "bottom-right");
        assert_eq!(apply(240.0, 135.0), (0.0, 0.0), "centre");
    }

    #[test]
    fn from_settings_ors_the_force_unicode_option() {
        let mut s = GameSettings {
            gui_scale: 0,
            force_unicode_font: true,
            ..GameSettings::default()
        };
        let r = ScaledResolution::from_settings(3840, 2160, &s, false);
        assert_eq!(r.scale_factor(), 8, "forceUnicodeFont alone is enough");
        s.force_unicode_font = false;
        assert_eq!(
            ScaledResolution::from_settings(3840, 2160, &s, false).scale_factor(),
            9
        );
        assert_eq!(
            ScaledResolution::from_settings(3840, 2160, &s, true).scale_factor(),
            8,
            "the language heuristic alone is also enough"
        );
    }

    /// The whole point of the scaled surface: it never gets smaller than 320x240 unless the
    /// window itself is.
    #[test]
    fn auto_scaling_never_goes_below_the_floor() {
        for w in [320u32, 640, 800, 1024, 1280, 1366, 1600, 1920, 2560, 3840] {
            for h in [240u32, 480, 600, 720, 768, 900, 1080, 1440, 2160] {
                let r = ScaledResolution::new(w, h, 0, false);
                assert!(
                    r.scaled_width() >= MIN_SCALED_WIDTH.min(w as i32),
                    "{w}x{h} -> {}",
                    r.scaled_width()
                );
                assert!(
                    r.scaled_height() >= MIN_SCALED_HEIGHT.min(h as i32),
                    "{w}x{h} -> {}",
                    r.scaled_height()
                );
                assert!(r.scale_factor() >= 1);
            }
        }
    }
}
