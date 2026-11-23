//! Marker pins in the world.
//!
//! A marker is a short-lived "look here": a pin in the marker's colour whose tip touches the
//! marked point, with the distance above it, readable through walls at a constant size on
//! screen:
//!
//! ```text
//!     ┌──────┐
//!     │ 42m  │    backdrop (vanilla's nameplate backdrop, black at alpha 0.25), white glyphs
//!     └──────┘
//!      ▜██▛       the pin: a downward triangle in the marker colour on a darker outline
//!       ▜▛
//!        ·        the marked point, under the pin's tip (the anchor)
//! ```
//!
//! # Geometry
//!
//! The waypoint label's: [`crate::waypoint_label::label_matrix`] (vanilla's nameplate billboard,
//! moved along its own view ray to at most 16 blocks so the far plane never clips it, and scaled
//! so it keeps one size on screen beyond 6 blocks). The two mods read as one family; the drawing
//! differs (a pin instead of a diamond, the distance alone instead of a name), so a marker is
//! never mistaken for a waypoint.
//!
//! # Drawing: [`MarkerLabel::emit`]
//!
//! Everything goes into [`NameplatePass::ThroughWalls`] (no depth test, no depth write) at vertex
//! alpha 1 for the pin and the text, so a marker reads at full strength through terrain.
//!
//! # Aiming: [`aimed_at`]
//!
//! Pressing the marker key on a marker removes it. "On" means the crosshair ray passes over the
//! pin or the distance box as drawn (plus [`AIM_MARGIN`]), so what counts is exactly what the
//! player sees, at any distance.
//!
//! # Cost
//!
//! [`MarkerLabel::update`] lays the text out only when the rounded distance or the font changed
//! (about once per block walked). Per frame: one matrix applied to 12 solid vertices plus 6 per
//! glyph, into reused buffers.

use glam::{DVec3, Mat4, Vec3};
use mc_render::entity::{argb_to_rgba, NameplatePass, NameplateVertex};

use crate::font::{FontRenderer, FontTexture};
use crate::nameplate::NameplateTarget;
use crate::waypoint_label::{
    distance_key, emit_glyph_runs, label_matrix, push_distance, push_quad, LabelBar, LabelGlyph,
    LabelScratch, Recorder, BACKDROP_RGBA,
};

/// Half the pin's width at its top edge, in label pixels (a 9-pixel-wide pin).
pub const PIN_HALF_WIDTH: f32 = 4.5;

/// The pin's height from its tip (the anchor) to its top edge, in label pixels.
pub const PIN_HEIGHT: f32 = 8.0;

/// How far the outline reaches past every edge of the pin, in label pixels.
pub const OUTLINE: f32 = 1.0;

/// The outline colour: black at alpha 0.55, so the pin stands out on bright sky or sand.
pub const OUTLINE_RGBA: [f32; 4] = [0.0, 0.0, 0.0, 0.55];

/// Label-local Y of the glyph row. The backdrop spans `TEXT_Y - 1 ..= TEXT_Y + 8`, so its bottom
/// edge sits 3 label pixels above the outline's top edge.
pub const TEXT_Y: f32 = -20.0;

/// The distance text colour: opaque white.
pub const TEXT_ARGB: u32 = 0xFFFF_FFFF;

/// Slack around the pin and the distance box for [`aimed_at`], in label pixels (about 0.8° of
/// view beyond 6 blocks, at 100% size).
pub const AIM_MARGIN: f32 = 3.0;

/// The pin's two triangles, `[tip, top-left, top-right]` in label pixels: the fill (tip on the
/// anchor) and the outline (the fill with every edge pushed out by [`OUTLINE`]).
fn pin_triangles() -> ([[f32; 2]; 3], [[f32; 2]; 3]) {
    let (w, h, o) = (PIN_HALF_WIDTH, PIN_HEIGHT, OUTLINE);
    let side = w.hypot(h);
    // Pushed out by `o`, the two slanted edges meet `o * side / w` below the old tip, and each
    // meets the raised top edge `o * (side + w) / h` beyond the old corner.
    let tip = o * side / w;
    let x = w + o * (side + w) / h;
    (
        [[0.0, 0.0], [-w, -h], [w, -h]],
        [[0.0, tip], [-x, -h - o], [x, -h - o]],
    )
}

/// One marker's cached distance text.
#[derive(Debug, Clone, Default)]
pub struct MarkerLabel {
    built: bool,
    distance: i64,
    font_generation: u64,
    text: String,
    width: i32,
    glyphs: Vec<LabelGlyph>,
    bars: Vec<LabelBar>,
    sheets: Vec<FontTexture>,
}

impl MarkerLabel {
    /// Lay out the distance text for `distance` (blocks): `42m`, or `12.3km` from 10 000
    /// blocks ([`push_distance`]).
    ///
    /// Does nothing unless the [`distance_key`] or [`FontRenderer::layout_generation`] changed.
    /// Returns whether it rebuilt.
    pub fn update(&mut self, font: &mut FontRenderer, distance: f64) -> bool {
        let key = distance_key(distance);
        let generation = font.layout_generation();
        if self.built && self.distance == key && self.font_generation == generation {
            return false;
        }
        self.built = true;
        self.distance = key;
        self.font_generation = generation;
        self.text.clear();
        push_distance(&mut self.text, key);
        self.width = font.string_width(&self.text);
        self.glyphs.clear();
        self.bars.clear();
        // Centred like a nameplate: `-getStringWidth(str) / 2`, integer division. No shadow: the
        // backdrop gives the contrast.
        font.draw_string(
            &mut Recorder {
                glyphs: &mut self.glyphs,
                bars: &mut self.bars,
            },
            &self.text,
            -((self.width / 2) as f32),
            TEXT_Y,
            TEXT_ARGB as i32,
            false,
        );
        self.sheets.clear();
        for glyph in &self.glyphs {
            if !self.sheets.contains(&glyph.texture) {
                self.sheets.push(glyph.texture);
            }
        }
        true
    }

    /// Whether [`Self::update`] has laid the text out since the last [`Self::invalidate`].
    #[must_use]
    pub fn is_built(&self) -> bool {
        self.built
    }

    /// The laid-out text. Empty before the first [`Self::update`].
    #[must_use]
    pub fn text(&self) -> &str {
        &self.text
    }

    /// The text's width in label pixels.
    #[must_use]
    pub fn width(&self) -> i32 {
        self.width
    }

    /// Every font sheet the glyphs sample, so the caller can make sure each is uploaded.
    #[must_use]
    pub fn sheets(&self) -> &[FontTexture] {
        &self.sheets
    }

    /// Forget the layout, so the next [`Self::update`] rebuilds it.
    pub fn invalidate(&mut self) {
        self.built = false;
    }

    /// Push this marker through `matrix` (from [`label_matrix`]) into `target`, all in
    /// [`NameplatePass::ThroughWalls`]: one solid run (the distance backdrop once laid out, the
    /// pin's outline, the pin in `rgb` = `0xRRGGBB`), then one glyph run per font sheet.
    pub fn emit<T: NameplateTarget + ?Sized>(
        &self,
        matrix: Mat4,
        rgb: u32,
        scratch: &mut LabelScratch,
        target: &mut T,
    ) {
        let place =
            |x: f32, y: f32| -> [f32; 3] { matrix.transform_point3(Vec3::new(x, y, 0.0)).into() };

        scratch.solid.clear();
        if self.built {
            // java: `renderLivingLabel`'s backdrop, `(-j - 1, -1)` to `(j + 1, 8)` around the text.
            let j = (self.width / 2) as f32;
            let (top, bottom) = (TEXT_Y - 1.0, TEXT_Y + 8.0);
            push_quad(
                &mut scratch.solid,
                [
                    place(-j - 1.0, top),
                    place(j + 1.0, top),
                    place(j + 1.0, bottom),
                    place(-j - 1.0, bottom),
                ],
                BACKDROP_RGBA,
            );
        }
        let (fill, outline) = pin_triangles();
        push_triangle(
            &mut scratch.solid,
            outline.map(|[x, y]| place(x, y)),
            OUTLINE_RGBA,
        );
        push_triangle(
            &mut scratch.solid,
            fill.map(|[x, y]| place(x, y)),
            argb_to_rgba(0xFF00_0000 | (rgb & 0xFF_FFFF)),
        );
        for bar in &self.bars {
            push_quad(
                &mut scratch.solid,
                [
                    place(bar.x0, bar.y0),
                    place(bar.x1, bar.y0),
                    place(bar.x1, bar.y1),
                    place(bar.x0, bar.y1),
                ],
                bar.rgba,
            );
        }
        target.push_solid(NameplatePass::ThroughWalls, &scratch.solid);
        emit_glyph_runs(&self.glyphs, place, &mut scratch.glyphs, target);
    }
}

/// One triangle, in the winding [`push_quad`] uses for its two.
fn push_triangle(out: &mut Vec<NameplateVertex>, corners: [[f32; 3]; 3], color: [f32; 4]) {
    for pos in corners {
        out.push(NameplateVertex { pos, color });
    }
}

/// Whether the crosshair is on a marker: the ray from the camera (the origin of these
/// camera-relative coordinates) along `look` crosses the marker's billboard over the pin or
/// over the distance box, each grown by [`AIM_MARGIN`].
///
/// `anchor_rel`, `yaw`, `pitch` and `scale` are what [`label_matrix`] takes (so the test follows
/// the drawn size at every distance); `text_width` is [`MarkerLabel::width`] (0 tests the pin
/// alone). `false` when the anchor is too close for a label, or behind the camera.
#[must_use]
pub fn aimed_at(
    anchor_rel: DVec3,
    look: DVec3,
    yaw: f32,
    pitch: f32,
    scale: f32,
    text_width: i32,
) -> bool {
    let Some(matrix) = label_matrix(anchor_rel, yaw, pitch, scale) else {
        return false;
    };
    // Into label pixels: the billboard is the plane z = 0 there.
    let inverse = matrix.inverse();
    let origin = inverse.transform_point3(Vec3::ZERO);
    let direction = inverse.transform_vector3(look.as_vec3());
    if !direction.z.is_finite() || direction.z.abs() < 1e-9 {
        return false;
    }
    let t = -origin.z / direction.z;
    if t.is_nan() || t <= 0.0 {
        return false;
    }
    let hit = origin + direction * t;
    let m = AIM_MARGIN;
    // The outline's corners: `[tip, top-left, top-right]`.
    let (_, [tip, _, top_right]) = pin_triangles();
    let on_pin =
        hit.x.abs() <= top_right[0] + m && (top_right[1] - m..=tip[1] + m).contains(&hit.y);
    // The backdrop reaches one pixel past the text on each side; no text, no box.
    let on_text = text_width > 0
        && hit.x.abs() <= (text_width / 2) as f32 + 1.0 + m
        && (TEXT_Y - 1.0 - m..=TEXT_Y + 8.0 + m).contains(&hit.y);
    on_pin || on_text
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::widget::test_support::uniform_font;
    use glam::Vec4;
    use mc_render::entity::NameplateGlyphVertex;
    use mc_render::Camera;

    #[derive(Default)]
    struct Recording {
        solid: Vec<(NameplatePass, Vec<NameplateVertex>)>,
        glyphs: Vec<(NameplatePass, FontTexture, Vec<NameplateGlyphVertex>)>,
    }

    impl NameplateTarget for Recording {
        fn push_solid(&mut self, pass: NameplatePass, verts: &[NameplateVertex]) {
            self.solid.push((pass, verts.to_vec()));
        }

        fn push_glyph_run(
            &mut self,
            pass: NameplatePass,
            texture: FontTexture,
            verts: &[NameplateGlyphVertex],
        ) {
            self.glyphs.push((pass, texture, verts.to_vec()));
        }
    }

    fn camera() -> Camera {
        let mut camera = Camera::default();
        camera.set_viewport(256, 256);
        camera.update();
        camera
    }

    /// Screen pixel (x, y) of a camera-relative world point.
    fn project(camera: &Camera, world: Vec3) -> (f32, f32) {
        let clip = camera.view_proj() * Vec4::new(world.x, world.y, world.z, 1.0);
        let ndc = clip.truncate() / clip.w;
        ((ndc.x * 0.5 + 0.5) * 256.0, (0.5 - ndc.y * 0.5) * 256.0)
    }

    /// The look vector from the camera through a label-local point of a marker at `rel`.
    fn look_through(rel: DVec3, yaw: f32, pitch: f32, x: f32, y: f32) -> DVec3 {
        let matrix = label_matrix(rel, yaw, pitch, 1.0).unwrap();
        matrix
            .transform_point3(Vec3::new(x, y, 0.0))
            .as_dvec3()
            .normalize()
    }

    #[test]
    fn the_text_is_the_distance_and_is_rebuilt_only_when_it_changes() {
        let mut font = uniform_font();
        let mut label = MarkerLabel::default();
        assert!(!label.is_built());
        assert!(label.update(&mut font, 48.2));
        assert_eq!(label.text(), "48m");
        assert_eq!(label.width(), font.string_width("48m"));
        assert!(!label.update(&mut font, 48.4), "still 48m");
        assert!(!label.update(&mut font, 47.6), "still 48m");
        assert!(label.update(&mut font, 47.4));
        assert_eq!(label.text(), "47m");
        assert!(label.update(&mut font, 12_345.0));
        assert_eq!(label.text(), "12.3km");
        label.invalidate();
        assert!(label.update(&mut font, 12_345.0), "invalidated");
        assert_eq!(label.sheets(), [FontTexture::Ascii]);
    }

    #[test]
    fn emit_pushes_backdrop_outline_pin_and_white_glyphs_through_walls() {
        let mut font = uniform_font();
        let mut label = MarkerLabel::default();
        label.update(&mut font, 42.0);
        let rel = DVec3::new(2.0, 1.0, 42.0);
        let matrix = label_matrix(rel, 0.0, 0.0, 1.0).unwrap();
        let mut recording = Recording::default();
        let mut scratch = LabelScratch::default();
        label.emit(matrix, 0x55_FFFF, &mut scratch, &mut recording);

        assert_eq!(recording.solid.len(), 1, "one solid run");
        let (pass, solid) = &recording.solid[0];
        assert_eq!(*pass, NameplatePass::ThroughWalls);
        assert_eq!(
            solid.len(),
            6 + 3 + 3,
            "backdrop quad, outline and pin triangles"
        );
        assert!(solid[..6].iter().all(|v| v.color == BACKDROP_RGBA));
        assert!(solid[6..9].iter().all(|v| v.color == OUTLINE_RGBA));
        let aqua = argb_to_rgba(0xFF55_FFFF);
        assert!(
            solid[9..].iter().all(|v| v.color == aqua),
            "opaque marker colour"
        );
        // The pin's tip is on the anchor (the proxy point on the anchor's ray).
        let tip = Vec3::from(solid[9].pos);
        assert!((tip - matrix.transform_point3(Vec3::ZERO)).length() < 1e-5);

        assert_eq!(recording.glyphs.len(), 1, "one sheet, one run");
        let (pass, texture, glyphs) = &recording.glyphs[0];
        assert_eq!(
            (*pass, *texture),
            (NameplatePass::ThroughWalls, FontTexture::Ascii)
        );
        assert_eq!(glyphs.len(), 6 * "42m".len());
        assert!(glyphs.iter().all(|v| v.color == argb_to_rgba(TEXT_ARGB)));

        // On screen: the text box above the outline, the outline above the tip, all centred.
        let camera = camera();
        let screen = |i: usize| project(&camera, Vec3::from(solid[i].pos));
        let (_, box_bottom) = screen(2);
        let (_, outline_top) = screen(7);
        let (tip_x, tip_y) = screen(9);
        assert!(box_bottom < outline_top && outline_top < tip_y);
        let (left_x, _) = screen(10);
        let (right_x, _) = screen(11);
        assert!(
            ((left_x + right_x) / 2.0 - tip_x).abs() < 1e-3,
            "centred over the tip"
        );
    }

    #[test]
    fn the_pin_keeps_its_size_on_screen_from_10_to_5000_blocks() {
        let camera = camera();
        let pin_height = |d: f64| {
            let matrix = label_matrix(DVec3::new(0.0, 0.0, d), 0.0, 0.0, 1.0).unwrap();
            let top = matrix.transform_point3(Vec3::new(0.0, -PIN_HEIGHT, 0.0));
            let (_, top) = project(&camera, top);
            let (_, tip) = project(&camera, matrix.transform_point3(Vec3::ZERO));
            tip - top
        };
        let near = pin_height(10.0);
        assert!(near > 3.0, "visible: {near} px");
        for d in [48.0, 256.0, 5000.0] {
            assert!((pin_height(d) - near).abs() < 1e-3, "{d} blocks");
        }
    }

    #[test]
    fn aiming_at_the_pin_or_the_distance_counts_and_beside_it_does_not() {
        let width = uniform_font().string_width("48m");
        for (yaw, pitch) in [(0.0, 0.0), (37.0, -12.0), (-150.0, 60.0)] {
            for distance in [3.0, 10.0, 48.0, 5000.0] {
                // A marker straight along the camera's forward-ish direction, off to one side.
                let forward = label_matrix(DVec3::Z, yaw, pitch, 1.0)
                    .unwrap()
                    .transform_vector3(Vec3::Z)
                    .as_dvec3()
                    .normalize();
                let rel = (forward + DVec3::new(0.05, -0.03, 0.02)).normalize() * distance;
                let at = |x: f32, y: f32| {
                    let look = look_through(rel, yaw, pitch, x, y);
                    aimed_at(rel, look, yaw, pitch, 1.0, width)
                };
                let case = format!("yaw {yaw}, pitch {pitch}, {distance} blocks");
                assert!(at(0.0, 0.0), "the tip: {case}");
                assert!(at(0.0, -PIN_HEIGHT / 2.0), "the pin: {case}");
                assert!(
                    at(PIN_HALF_WIDTH + 2.0, -PIN_HEIGHT),
                    "inside the margin: {case}"
                );
                assert!(at(0.0, TEXT_Y + 4.0), "the distance: {case}");
                assert!(
                    at(-(width as f32) / 2.0, TEXT_Y + 4.0),
                    "its left end: {case}"
                );
                assert!(!at(20.0, -PIN_HEIGHT / 2.0), "beside the pin: {case}");
                assert!(!at(0.0, 10.0), "below the tip: {case}");
                assert!(!at(0.0, TEXT_Y - 10.0), "above the box: {case}");
                assert!(!at(30.0, TEXT_Y + 4.0), "beside the box: {case}");
                // Looking the opposite way never counts, whatever the plane maths says.
                let back = -look_through(rel, yaw, pitch, 0.0, 0.0);
                assert!(
                    !aimed_at(rel, back, yaw, pitch, 1.0, width),
                    "behind: {case}"
                );
            }
        }
        // Without a laid-out text only the pin counts.
        let rel = DVec3::new(0.0, 0.0, 48.0);
        let at_text = look_through(rel, 0.0, 0.0, 0.0, TEXT_Y + 4.0);
        assert!(!aimed_at(rel, at_text, 0.0, 0.0, 1.0, 0));
        let at_pin = look_through(rel, 0.0, 0.0, 0.0, -1.0);
        assert!(aimed_at(rel, at_pin, 0.0, 0.0, 1.0, 0));
        // Inside the marked spot there is no label, so nothing to aim at.
        assert!(!aimed_at(
            DVec3::new(0.0, 0.0, 0.1),
            DVec3::Z,
            0.0,
            0.0,
            1.0,
            width
        ));
    }

    #[test]
    fn the_outline_surrounds_the_pin_by_its_width_on_every_edge() {
        let (fill, outline) = pin_triangles();
        // Distance from a point to the line through a and b.
        let line_distance = |p: [f32; 2], a: [f32; 2], b: [f32; 2]| {
            let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
            ((p[0] - a[0]) * dy - (p[1] - a[1]) * dx).abs() / dx.hypot(dy)
        };
        for i in 0..3 {
            let (a, b) = (fill[i], fill[(i + 1) % 3]);
            let (oa, ob) = (outline[i], outline[(i + 1) % 3]);
            assert!((line_distance(oa, a, b) - OUTLINE).abs() < 1e-4, "edge {i}");
            assert!((line_distance(ob, a, b) - OUTLINE).abs() < 1e-4, "edge {i}");
        }
    }
}
