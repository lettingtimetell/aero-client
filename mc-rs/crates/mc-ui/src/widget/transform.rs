//! A 2-D affine transform, and the matrix stack [`DrawList`](crate::widget::DrawList) applies it
//! with.
//!
//! java: `client/renderer/GlStateManager.java` (`pushMatrix`/`translate`/`rotate`/`scale`/
//! `popMatrix`), as used by `GuiMainMenu.drawScreen`, `GuiIngame.renderTooltip` and
//! `GuiNewChat.drawChat`
//!
//! # Why a stack and not a per-batch matrix
//!
//! Every place 1.8.9's GUI scales or rotates does the same thing: push the GL matrix, translate
//! to an anchor, rotate and/or scale, run an *otherwise completely ordinary* sequence of draws,
//! pop. Three examples, and they are the whole set:
//!
//! | Where | Java | Ops |
//! |---|---|---|
//! | Main-menu splash | `GuiMainMenu.java:560` | `translate(w/2 + 90, 70)` → `rotate(-20°)` → `scale(f)` |
//! | `/title` + subtitle | `GuiIngame.java:298` | `translate(w/2, h/2)` → `scale(4)` … `scale(2)` |
//! | Chat box | `GuiNewChat.java:46` | `translate(2, 20)` → `scale(chatScale)` |
//!
//! The title case nests: one outer `translate`, then two independent inner `scale`s, each with
//! its own push/pop. A stack reproduces that directly; a single affine per [`DrawBatch`] would
//! not, because the two scales interleave with draws that share a texture and would therefore
//! have to *split* a batch — changing a frozen public type and the shader that consumes it.
//!
//! [`DrawBatch`]: crate::widget::DrawBatch
//!
//! # Where the transform is applied
//!
//! At vertex-emission time, to positions only, and **only while the stack is non-empty**. That
//! is the same place GL applies the modelview matrix: after the caller has computed integer
//! pixel coordinates, before the projection. In particular
//! [`FontRenderer`](crate::font::FontRenderer) still lays glyphs out on whole pixels and still
//! advances by the `glyph_sizes.bin` widths (hazard **H9**) — a fractional scale multiplies the
//! finished layout and cannot re-round an advance, exactly as in vanilla, where a scaled string
//! is a scaled *picture* of the unscaled one.
//!
//! # Interaction with `ScaledResolution`
//!
//! This composes *inside* the GUI scale, not with it:
//! [`ScaledResolution::ortho`](crate::widget::ScaledResolution::ortho) is still the projection
//! and still maps GUI pixels to clip space. A `scale(2)` pushed here at `guiScale = 2` therefore
//! turns one texel into a 4×4 block of window pixels, which is what vanilla does too.

/// A 2×3 affine transform over GUI pixels.
///
/// Row-major, with an implicit `[0, 0, 1]` third row:
///
/// ```text
/// x' = m[0][0] * x + m[0][1] * y + m[0][2]
/// y' = m[1][0] * x + m[1][1] * y + m[1][2]
/// ```
///
/// Build one the way you would write the `GlStateManager` calls, left to right:
///
/// ```
/// use mc_ui::widget::Transform;
/// // GuiMainMenu: translate(x, 70) -> rotate(-20) -> scale(1.8)
/// let t = Transform::translate(100.0, 70.0)
///     .then_rotate_degrees(-20.0)
///     .then_scale(1.8);
/// let [x, y] = t.apply([0.0, 0.0]);
/// assert!((x - 100.0).abs() < 1e-4 && (y - 70.0).abs() < 1e-4);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Transform {
    /// The two rows of the matrix, `[[xx, xy, dx], [yx, yy, dy]]`.
    pub m: [[f32; 3]; 2],
}

impl Default for Transform {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Transform {
    /// The transform that changes nothing.
    pub const IDENTITY: Self = Self {
        m: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    };

    /// A pure translation.
    // java: GlStateManager.translate(x, y, 0)
    #[must_use]
    pub const fn translate(dx: f32, dy: f32) -> Self {
        Self {
            m: [[1.0, 0.0, dx], [0.0, 1.0, dy]],
        }
    }

    /// A uniform scale about the origin, vanilla's `scale(f, f, f)`.
    // java: GlStateManager.scale(f, f, f)
    #[must_use]
    pub const fn scale(f: f32) -> Self {
        Self::scale_xy(f, f)
    }

    /// A non-uniform scale about the origin.
    ///
    /// `GuiIngame`'s damage-tilt hearts use one (`scale(1/f, (f+1)/2, 1)`), so the general form
    /// is worth having even though the three headline cases are uniform.
    // java: GlStateManager.scale(x, y, 1)
    #[must_use]
    pub const fn scale_xy(sx: f32, sy: f32) -> Self {
        Self {
            m: [[sx, 0.0, 0.0], [0.0, sy, 0.0]],
        }
    }

    /// A rotation about the origin, in degrees, matching `glRotatef(deg, 0, 0, 1)`.
    ///
    /// Positive angles turn *clockwise* on screen, because the GUI projection has `y` growing
    /// downwards — which is why the splash's `-20°` tilts up to the right.
    ///
    /// The sine and cosine come from the standard library in `f64` and not from
    /// [`mc_math::math_helper::sin`], deliberately: hazard **H1**'s 65536-entry LUT exists to
    /// match Java's *game logic*, whereas `glRotatef` is evaluated by the GL driver in full
    /// precision. Using the LUT here would put a visible kink in the splash.
    // java: GlStateManager.rotate(angle, 0.0F, 0.0F, 1.0F)
    #[must_use]
    pub fn rotate_degrees(degrees: f32) -> Self {
        let r = f64::from(degrees).to_radians();
        let (sin, cos) = (r.sin() as f32, r.cos() as f32);
        Self {
            m: [[cos, -sin, 0.0], [sin, cos, 0.0]],
        }
    }

    /// `self * local` — apply `local` inside `self`'s own coordinate space.
    ///
    /// This is exactly what a further `GlStateManager` call does to the current matrix: GL
    /// right-multiplies, so the transform written *last* is the one applied to a vertex
    /// *first*. Prefer the `then_*` helpers, which read in GL-call order.
    #[must_use]
    pub fn concat(self, local: Self) -> Self {
        let a = self.m;
        let b = local.m;
        Self {
            m: [
                [
                    a[0][0] * b[0][0] + a[0][1] * b[1][0],
                    a[0][0] * b[0][1] + a[0][1] * b[1][1],
                    a[0][0] * b[0][2] + a[0][1] * b[1][2] + a[0][2],
                ],
                [
                    a[1][0] * b[0][0] + a[1][1] * b[1][0],
                    a[1][0] * b[0][1] + a[1][1] * b[1][1],
                    a[1][0] * b[0][2] + a[1][1] * b[1][2] + a[1][2],
                ],
            ],
        }
    }

    /// [`concat`](Self::concat) a translation — the next `GlStateManager.translate`.
    #[must_use]
    pub fn then_translate(self, dx: f32, dy: f32) -> Self {
        self.concat(Self::translate(dx, dy))
    }

    /// [`concat`](Self::concat) a uniform scale — the next `GlStateManager.scale`.
    #[must_use]
    pub fn then_scale(self, f: f32) -> Self {
        self.concat(Self::scale(f))
    }

    /// [`concat`](Self::concat) a non-uniform scale.
    #[must_use]
    pub fn then_scale_xy(self, sx: f32, sy: f32) -> Self {
        self.concat(Self::scale_xy(sx, sy))
    }

    /// [`concat`](Self::concat) a rotation — the next `GlStateManager.rotate`.
    #[must_use]
    pub fn then_rotate_degrees(self, degrees: f32) -> Self {
        self.concat(Self::rotate_degrees(degrees))
    }

    /// Transform one point.
    #[must_use]
    pub fn apply(self, [x, y]: [f32; 2]) -> [f32; 2] {
        [
            self.m[0][0] * x + self.m[0][1] * y + self.m[0][2],
            self.m[1][0] * x + self.m[1][1] * y + self.m[1][2],
        ]
    }

    /// The 2×2 determinant. Zero exactly when the transform collapses the plane.
    #[must_use]
    pub fn determinant(self) -> f32 {
        self.m[0][0] * self.m[1][1] - self.m[0][1] * self.m[1][0]
    }

    /// The inverse, or `None` when the transform is singular.
    ///
    /// This is what a screen needs to hit-test inside its own scaled content — vanilla writes
    /// the same thing by hand as `(int)((float)mouseX / chatScale)` in
    /// `GuiNewChat.getChatComponent`.
    #[must_use]
    pub fn try_inverse(self) -> Option<Self> {
        let det = self.determinant();
        if det == 0.0 || !det.is_finite() {
            return None;
        }
        let [[a, b, c], [d, e, f]] = self.m;
        let inv = 1.0 / det;
        Some(Self {
            m: [
                [e * inv, -b * inv, (b * f - e * c) * inv],
                [-d * inv, a * inv, (d * c - a * f) * inv],
            ],
        })
    }

    /// Whether this is bit-for-bit the identity.
    #[must_use]
    pub fn is_identity(self) -> bool {
        self.m == Self::IDENTITY.m
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: [f32; 2], b: [f32; 2]) -> bool {
        (a[0] - b[0]).abs() < 1e-3 && (a[1] - b[1]).abs() < 1e-3
    }

    #[test]
    fn the_identity_moves_nothing() {
        let t = Transform::IDENTITY;
        assert!(t.is_identity());
        assert_eq!(t.apply([3.5, -7.25]), [3.5, -7.25]);
        assert_eq!(Transform::default(), Transform::IDENTITY);
    }

    #[test]
    fn translate_scale_and_rotate_do_what_they_say() {
        assert_eq!(
            Transform::translate(10.0, -4.0).apply([1.0, 1.0]),
            [11.0, -3.0]
        );
        assert_eq!(Transform::scale(3.0).apply([2.0, 5.0]), [6.0, 15.0]);
        assert_eq!(Transform::scale_xy(2.0, 0.5).apply([3.0, 8.0]), [6.0, 4.0]);
        // 90 degrees about +Z: (1, 0) -> (0, 1), which with y-down is a clockwise quarter turn.
        assert!(close(
            Transform::rotate_degrees(90.0).apply([1.0, 0.0]),
            [0.0, 1.0]
        ));
        assert!(close(
            Transform::rotate_degrees(-90.0).apply([1.0, 0.0]),
            [0.0, -1.0]
        ));
        assert!(close(
            Transform::rotate_degrees(180.0).apply([2.0, 3.0]),
            [-2.0, -3.0]
        ));
    }

    /// `concat` has to be GL's right-multiply, or a translate-then-scale silently becomes a
    /// scale-then-translate and every anchored, scaled screen lands in the wrong place.
    #[test]
    fn concatenation_is_gls_right_multiply() {
        let t = Transform::translate(100.0, 50.0).then_scale(2.0);
        // The anchor is untouched by the scale that follows it...
        assert_eq!(t.apply([0.0, 0.0]), [100.0, 50.0]);
        // ...and offsets from the anchor are doubled.
        assert_eq!(t.apply([10.0, 10.0]), [120.0, 70.0]);

        // The other order scales the translation itself.
        let u = Transform::scale(2.0).then_translate(100.0, 50.0);
        assert_eq!(u.apply([0.0, 0.0]), [200.0, 100.0]);
        assert_ne!(t, u);
    }

    /// The exact `GuiMainMenu` splash sequence: the text is centred on the anchor whatever the
    /// scale, and the tilt is up-and-to-the-right.
    // java: GuiMainMenu.java:560
    #[test]
    fn the_splash_transform_anchors_then_tilts_then_scales() {
        let f = 1.8f32;
        let t = Transform::translate(340.0, 70.0)
            .then_rotate_degrees(-20.0)
            .then_scale(f);
        assert!(
            close(t.apply([0.0, 0.0]), [340.0, 70.0]),
            "the anchor is fixed"
        );
        // drawCenteredString(..., 0, -8, ...) puts the text 8 px above the anchor in local
        // space; -20 degrees lifts its right-hand end.
        let left = t.apply([-30.0, -8.0]);
        let right = t.apply([30.0, -8.0]);
        assert!(right[0] > left[0], "the string still runs left to right");
        assert!(right[1] < left[1], "and its right-hand end is higher up");
        // The tilt is the same 20 degrees regardless of f.
        let dy = (right[1] - left[1]) / (right[0] - left[0]);
        assert!((dy + (20.0f32).to_radians().tan()).abs() < 1e-4, "{dy}");
    }

    #[test]
    fn nested_pushes_compose_the_way_the_title_screen_nests_them() {
        // GuiIngame: translate(w/2, h/2), then two independent inner scales.
        let outer = Transform::translate(200.0, 120.0);
        let title = outer.then_scale(4.0);
        let subtitle = outer.then_scale(2.0);
        assert_eq!(title.apply([-10.0, -10.0]), [160.0, 80.0]);
        assert_eq!(subtitle.apply([-10.0, 5.0]), [180.0, 130.0]);
        // Popping back to `outer` must leave it untouched — it is a value, not a mutation.
        assert_eq!(outer.apply([1.0, 1.0]), [201.0, 121.0]);
    }

    #[test]
    fn the_inverse_maps_a_cursor_back_into_scaled_content() {
        // GuiNewChat: translate(2, 20), scale(chatScale).
        let t = Transform::translate(2.0, 20.0).then_scale(0.5);
        let inv = t.try_inverse().expect("invertible");
        let screen = t.apply([40.0, -18.0]);
        assert!(close(inv.apply(screen), [40.0, -18.0]));
        // And a rotation round-trips too.
        let r = Transform::translate(5.0, 6.0)
            .then_rotate_degrees(-20.0)
            .then_scale(1.75);
        let ri = r.try_inverse().expect("invertible");
        assert!(close(ri.apply(r.apply([13.0, -4.0])), [13.0, -4.0]));
    }

    #[test]
    fn a_collapsed_transform_has_no_inverse() {
        assert_eq!(Transform::scale(0.0).determinant(), 0.0);
        assert!(Transform::scale(0.0).try_inverse().is_none());
        assert!(Transform::scale_xy(1.0, 0.0).try_inverse().is_none());
        assert!(Transform::IDENTITY.try_inverse().is_some());
    }

    /// `glRotatef` is evaluated by the driver, not by `MathHelper`'s LUT — so a rotation this
    /// port produces has to agree with the real trig function, not with the 65536-entry table.
    #[test]
    fn rotation_uses_real_trigonometry_not_the_math_helper_lut() {
        let deg = -20.0f32;
        let t = Transform::rotate_degrees(deg);
        let rad = f64::from(deg).to_radians();
        assert_eq!(t.m[0][0], rad.cos() as f32);
        assert_eq!(t.m[1][0], rad.sin() as f32);
        // The LUT is coarser than this by more than a float epsilon at 20 degrees.
        let lut = mc_math::math_helper::sin(rad as f32);
        assert!(
            (lut - t.m[1][0]).abs() > 1e-6,
            "the LUT and libm agree too closely for this test to mean anything: {lut}"
        );
    }
}
