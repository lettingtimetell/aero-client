//! Waypoint labels in the world.
//!
//! A waypoint shows its name and distance, readable through walls at a constant size on screen,
//! with a small diamond in the waypoint's colour on the exact spot and the label stacked above
//! it:
//!
//! ```text
//!     ┌──────────────┐
//!     │ Base (123m)  │   backdrop (black, alpha 0.25) and glyphs
//!     └──────────────┘
//!            ◆           5-pixel diamond on the block centre (the anchor)
//! ```
//!
//! # Geometry: [`label_matrix`]
//!
//! It is vanilla's nameplate billboard (`Render.renderLivingLabel`: translate, `rotY(-yaw)`,
//! `rotX(pitch)`, `scale(-k, -k, k)`), with two changes:
//!
//! * **Proxy distance.** The label is moved along its own view ray to at most
//!   [`PROXY_DISTANCE`] blocks, which is inside every far plane (render distance 2 gives about
//!   45 blocks). The scale grows with the same factor, and the billboard is parallel to the image
//!   plane, so the picture is identical; only far-plane clipping is avoided. Nothing depth-tests
//!   it, so moving it is invisible.
//! * **Constant size.** `k = NAMEPLATE_SCALE * scale * L / clamp(d, 2, 6)`, with `d` the true
//!   distance and `L` the proxy distance. Beyond 6 blocks the label keeps the on-screen size a
//!   vanilla nameplate has at 6 blocks; from 6 to 2 blocks it grows like a real nameplate; under
//!   2 blocks it stops growing at 3x, so standing on a waypoint never fills the screen.
//!
//! # Drawing: [`WaypointLabel::emit`]
//!
//! Everything goes into [`NameplatePass::ThroughWalls`] (no depth test, no depth write) with
//! vertex alpha 1 for the text and the marker, so a waypoint reads at full strength through
//! terrain. Pushes go through [`NameplateTarget`], so the production `NameplatePhase` and a test
//! recorder see the same stream.
//!
//! # Cost
//!
//! [`WaypointLabel::update`] lays the text out only when the name, the colour, the rounded
//! distance or the font changed. Per frame, only the billboard matrix is applied to the cached
//! label-local quads: about 6 vertices per glyph plus 12 for the backdrop and marker.

use std::fmt::Write as _;

use glam::{DVec3, Mat4, Vec3};
use mc_render::entity::nameplate::NAMEPLATE_SCALE;
use mc_render::entity::{argb_to_rgba, NameplateGlyphVertex, NameplatePass, NameplateVertex};

use crate::font::{FontRenderer, FontTexture, GlyphQuad, SolidRect, TextSink};
use crate::nameplate::NameplateTarget;

/// Closer than this (blocks) there is no label: the camera is inside the marked block.
pub const MIN_DISTANCE: f64 = 0.3;

/// The furthest a label is placed along its view ray, in blocks. Inside every far plane.
pub const PROXY_DISTANCE: f64 = 16.0;

/// Beyond this distance (blocks) a label keeps a constant size on screen: a vanilla nameplate's
/// size at this distance.
pub const CONSTANT_FROM: f64 = 6.0;

/// Closer than this (blocks) a label stops growing: `CONSTANT_FROM / GROWTH_STOP` = 3x its far
/// size.
pub const GROWTH_STOP: f64 = 2.0;

/// Half the marker diamond's width and height, in label pixels (a 5-pixel diamond).
pub const MARKER_HALF: f32 = 2.5;

/// Label-local Y of the glyph row. The backdrop spans `TEXT_Y - 1 ..= TEXT_Y + 8`, so its bottom
/// edge sits 2.5 label pixels above the marker's top point.
pub const TEXT_Y: f32 = -13.0;

/// The backdrop colour: vanilla's nameplate backdrop, black at alpha 0.25.
pub const BACKDROP_RGBA: [f32; 4] = [0.0, 0.0, 0.0, 0.25];

/// Rounded distances from this many blocks up are shown in kilometres.
pub const KM_FROM: i64 = 10_000;

/// The billboard matrix for a label whose anchor is `anchor_rel` (camera-relative, blocks),
/// seen with the camera's `yaw`/`pitch` (degrees), at `scale` (1.0 = 100%).
///
/// `None` when the anchor is closer than [`MIN_DISTANCE`] or not finite. See the module docs
/// for the proxy distance and the size rule.
#[must_use]
pub fn label_matrix(anchor_rel: DVec3, yaw: f32, pitch: f32, scale: f32) -> Option<Mat4> {
    let distance = anchor_rel.length();
    // `is_finite` is false for NaN as well as for the infinities.
    if !distance.is_finite() || distance < MIN_DISTANCE {
        return None;
    }
    let proxy = distance.min(PROXY_DISTANCE);
    let position = (anchor_rel * (proxy / distance)).as_vec3();
    let k = NAMEPLATE_SCALE * scale * (proxy / distance.clamp(GROWTH_STOP, CONSTANT_FROM)) as f32;
    Some(
        Mat4::from_translation(position)
            * Mat4::from_rotation_y((-yaw).to_radians())
            * Mat4::from_rotation_x(pitch.to_radians())
            * Mat4::from_scale(Vec3::new(-k, -k, k)),
    )
}

/// The value a label's distance text is built from: whole blocks below [`KM_FROM`], otherwise
/// `KM_FROM` plus tenths of a kilometre. Two distances with the same key have the same text.
#[must_use]
pub fn distance_key(distance: f64) -> i64 {
    let blocks = distance.max(0.0).round();
    if blocks < KM_FROM as f64 {
        blocks as i64
    } else {
        KM_FROM + (distance / 100.0).round() as i64
    }
}

/// Append the distance text for a [`distance_key`]: `123m`, or `12.3km` from 10 000 blocks.
pub fn push_distance(out: &mut String, key: i64) {
    if key < KM_FROM {
        let _ = write!(out, "{key}m");
    } else {
        let tenths = key - KM_FROM;
        let _ = write!(out, "{}.{}km", tenths / 10, tenths % 10);
    }
}

/// One glyph of a laid-out label, in label pixels, as the font emitted it.
///
/// `pub(crate)` (with [`LabelBar`], [`Recorder`], [`LabelScratch`]'s buffers, [`push_quad`] and
/// [`emit_glyph_runs`]) so `crate::marker_label`lays out and draws the same way.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct LabelGlyph {
    pub(crate) texture: FontTexture,
    pub(crate) pos: [[f32; 2]; 4],
    pub(crate) uv: [[f32; 2]; 4],
    pub(crate) rgba: [f32; 4],
}

/// A `§m`/`§n` bar of a laid-out label.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct LabelBar {
    pub(crate) x0: f32,
    pub(crate) y0: f32,
    pub(crate) x1: f32,
    pub(crate) y1: f32,
    pub(crate) rgba: [f32; 4],
}

/// Records the font's output as label-local geometry.
pub(crate) struct Recorder<'a> {
    pub(crate) glyphs: &'a mut Vec<LabelGlyph>,
    pub(crate) bars: &'a mut Vec<LabelBar>,
}

impl TextSink for Recorder<'_> {
    fn glyph(&mut self, quad: GlyphQuad) {
        self.glyphs.push(LabelGlyph {
            texture: quad.texture,
            pos: quad.pos,
            uv: quad.uv,
            rgba: argb_to_rgba(quad.color),
        });
    }

    fn bar(&mut self, rect: SolidRect) {
        self.bars.push(LabelBar {
            x0: rect.x0,
            y0: rect.y0,
            x1: rect.x1,
            y1: rect.y1,
            rgba: argb_to_rgba(rect.color),
        });
    }
}

/// Reusable vertex buffers for [`WaypointLabel::emit`], shared by every label of a frame.
#[derive(Debug, Default)]
pub struct LabelScratch {
    pub(crate) solid: Vec<NameplateVertex>,
    pub(crate) glyphs: Vec<NameplateGlyphVertex>,
}

/// One waypoint's cached label layout.
#[derive(Debug, Clone, Default)]
pub struct WaypointLabel {
    built: bool,
    name: String,
    rgb: u32,
    distance: Option<i64>,
    font_generation: u64,
    text: String,
    width: i32,
    glyphs: Vec<LabelGlyph>,
    bars: Vec<LabelBar>,
    sheets: Vec<FontTexture>,
}

impl WaypointLabel {
    /// Lay the label out for `name` in colour `rgb` (`0xRRGGBB`), followed by the distance in
    /// grey when `distance` (blocks) is given: `"{name}§7 ({n}m)"`.
    ///
    /// Does nothing unless the name, the colour, the [`distance_key`] or
    /// [`FontRenderer::layout_generation`] changed. Returns whether it rebuilt.
    pub fn update(
        &mut self,
        font: &mut FontRenderer,
        name: &str,
        rgb: u32,
        distance: Option<f64>,
    ) -> bool {
        let rgb = rgb & 0xFF_FFFF;
        let distance = distance.map(distance_key);
        let generation = font.layout_generation();
        if self.built
            && self.rgb == rgb
            && self.distance == distance
            && self.font_generation == generation
            && self.name == name
        {
            return false;
        }
        self.built = true;
        self.rgb = rgb;
        self.distance = distance;
        self.font_generation = generation;
        self.name.clear();
        self.name.push_str(name);
        self.text.clear();
        self.text.push_str(name);
        if let Some(key) = distance {
            self.text.push_str("\u{a7}7 (");
            push_distance(&mut self.text, key);
            self.text.push(')');
        }
        self.width = font.string_width(&self.text);
        self.glyphs.clear();
        self.bars.clear();
        // java: `drawString(str, -getStringWidth(str) / 2, y, colour)`, integer division, as
        // vanilla centres a nameplate. No shadow: the backdrop gives the contrast.
        font.draw_string(
            &mut Recorder {
                glyphs: &mut self.glyphs,
                bars: &mut self.bars,
            },
            &self.text,
            -((self.width / 2) as f32),
            TEXT_Y,
            (0xFF00_0000 | rgb) as i32,
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

    /// The laid-out text, `§` codes included. Empty before the first [`Self::update`].
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

    /// Push this label through `matrix` (from [`label_matrix`]) into `target`, all in
    /// [`NameplatePass::ThroughWalls`]: one solid run (backdrop, marker diamond, any `§m`/`§n`
    /// bars), then one glyph run per font sheet.
    pub fn emit<T: NameplateTarget + ?Sized>(
        &self,
        matrix: Mat4,
        scratch: &mut LabelScratch,
        target: &mut T,
    ) {
        let place =
            |x: f32, y: f32| -> [f32; 3] { matrix.transform_point3(Vec3::new(x, y, 0.0)).into() };

        scratch.solid.clear();
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
        let h = MARKER_HALF;
        push_quad(
            &mut scratch.solid,
            [place(0.0, -h), place(h, 0.0), place(0.0, h), place(-h, 0.0)],
            argb_to_rgba(0xFF00_0000 | self.rgb),
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

/// Push laid-out `glyphs` through `place` (label pixels to camera-relative world) into
/// `target`, in [`NameplatePass::ThroughWalls`]: one glyph run per consecutive font sheet.
/// `scratch` is reused for each run.
pub(crate) fn emit_glyph_runs<T: NameplateTarget + ?Sized>(
    glyphs: &[LabelGlyph],
    place: impl Fn(f32, f32) -> [f32; 3],
    scratch: &mut Vec<NameplateGlyphVertex>,
    target: &mut T,
) {
    // GL_TRIANGLE_STRIP over [0,1,2,3] is triangles (0,1,2) and (2,1,3), as in
    // `Nameplate::glyph_vertices_with_matrix`.
    const STRIP: [usize; 6] = [0, 1, 2, 2, 1, 3];
    let mut i = 0;
    while i < glyphs.len() {
        let texture = glyphs[i].texture;
        scratch.clear();
        while i < glyphs.len() && glyphs[i].texture == texture {
            let glyph = &glyphs[i];
            let corners = glyph.pos.map(|[x, y]| place(x, y));
            scratch.extend(STRIP.map(|k| NameplateGlyphVertex {
                pos: corners[k],
                uv: glyph.uv[k],
                color: glyph.rgba,
            }));
            i += 1;
        }
        target.push_glyph_run(NameplatePass::ThroughWalls, texture, scratch);
    }
}

/// Two triangles (0-1-2, 0-2-3) over four corners.
pub(crate) fn push_quad(out: &mut Vec<NameplateVertex>, corners: [[f32; 3]; 4], color: [f32; 4]) {
    for i in [0usize, 1, 2, 0, 2, 3] {
        out.push(NameplateVertex {
            pos: corners[i],
            color,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::widget::test_support::uniform_font;
    use glam::Vec4;
    use mc_render::Camera;

    /// Records what reaches a target, in push order.
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

    /// Screen pixel (x, y) and clip depth of a label-local point.
    fn project(camera: &Camera, matrix: Mat4, x: f32, y: f32) -> (f32, f32, f32) {
        let world = matrix.transform_point3(Vec3::new(x, y, 0.0));
        let clip = camera.view_proj() * Vec4::new(world.x, world.y, world.z, 1.0);
        let ndc = clip.truncate() / clip.w;
        (
            (ndc.x * 0.5 + 0.5) * 256.0,
            (0.5 - ndc.y * 0.5) * 256.0,
            ndc.z,
        )
    }

    /// On-screen height in pixels of the backdrop of a label anchored at `rel`.
    fn backdrop_height(camera: &Camera, rel: DVec3) -> f32 {
        let m = label_matrix(rel, camera.yaw, camera.pitch, 1.0).expect("far enough");
        let (_, top, _) = project(camera, m, 0.0, TEXT_Y - 1.0);
        let (_, bottom, _) = project(camera, m, 0.0, TEXT_Y + 8.0);
        bottom - top
    }

    #[test]
    fn a_label_is_the_same_size_on_screen_at_48_and_5000_blocks() {
        let camera = camera();
        // Yaw 0 looks along +Z. One anchor straight ahead, one off to the side.
        for direction in [DVec3::Z, DVec3::new(0.3, -0.2, 1.0).normalize()] {
            // 10 blocks is drawn where it is; 48 and 5000 are pulled in to the proxy distance.
            let real = backdrop_height(&camera, direction * 10.0);
            let near = backdrop_height(&camera, direction * 48.0);
            let far = backdrop_height(&camera, direction * 5000.0);
            assert!(near > 4.0, "the label is visible: {near}");
            assert!(
                (near - far).abs() < 1e-3,
                "48 blocks: {near} px, 5000 blocks: {far} px"
            );
            assert!(
                (near - real).abs() < 1e-3,
                "48 blocks: {near} px, 10 blocks: {real} px"
            );
            // Same spot on screen too, and inside the depth range (not far-clipped).
            let m48 = label_matrix(direction * 48.0, 0.0, 0.0, 1.0).unwrap();
            let m5000 = label_matrix(direction * 5000.0, 0.0, 0.0, 1.0).unwrap();
            let (x0, y0, _) = project(&camera, m48, 0.0, 0.0);
            let (x1, y1, z1) = project(&camera, m5000, 0.0, 0.0);
            assert!((x0 - x1).abs() < 1e-3 && (y0 - y1).abs() < 1e-3);
            assert!((0.0..1.0).contains(&z1), "inside the far plane: {z1}");
        }
    }

    #[test]
    fn the_proxy_sits_on_the_anchor_ray_at_most_16_blocks_out() {
        for rel in [
            DVec3::new(3.0, 4.0, 0.0),
            DVec3::new(-100.0, 20.0, 300.0),
            DVec3::new(0.0, -9000.0, 1.0),
        ] {
            let m = label_matrix(rel, 37.0, -12.0, 1.0).unwrap();
            let proxy = m.transform_point3(Vec3::ZERO).as_dvec3();
            let expected = rel.length().min(PROXY_DISTANCE);
            assert!(
                (proxy.length() - expected).abs() < 1e-3,
                "{proxy} vs {expected}"
            );
            assert!(proxy.normalize().dot(rel.normalize()) > 0.999_999);
        }
    }

    #[test]
    fn close_labels_grow_like_a_nameplate_then_stop_at_three_times() {
        let camera = camera();
        let at = |d: f64| backdrop_height(&camera, DVec3::Z * d);
        let far = at(48.0);
        assert!((at(6.0) - far).abs() < 1e-3);
        assert!(
            (at(3.0) - 2.0 * far).abs() < 1e-2,
            "{} vs {}",
            at(3.0),
            2.0 * far
        );
        assert!((at(2.0) - 3.0 * far).abs() < 1e-2);
        assert!((at(1.0) - 3.0 * far).abs() < 1e-2, "capped at 3x");
        // The label scale option multiplies everything.
        let m = label_matrix(DVec3::Z * 48.0, 0.0, 0.0, 2.0).unwrap();
        let (_, top, _) = project(&camera, m, 0.0, TEXT_Y - 1.0);
        let (_, bottom, _) = project(&camera, m, 0.0, TEXT_Y + 8.0);
        assert!(((bottom - top) - 2.0 * far).abs() < 1e-2);
        // At 6 blocks and beyond it is exactly a vanilla nameplate at 6 blocks.
        let nameplate = 9.0 * NAMEPLATE_SCALE / 6.0 * camera.proj.y_axis.y * 128.0;
        assert!((far - nameplate).abs() < 1e-2, "{far} vs {nameplate}");
    }

    #[test]
    fn there_is_no_label_inside_the_block_or_for_a_broken_position() {
        assert!(label_matrix(DVec3::new(0.1, 0.1, 0.1), 0.0, 0.0, 1.0).is_none());
        assert!(label_matrix(DVec3::splat(f64::NAN), 0.0, 0.0, 1.0).is_none());
        assert!(label_matrix(DVec3::new(f64::INFINITY, 0.0, 0.0), 0.0, 0.0, 1.0).is_none());
        assert!(label_matrix(DVec3::new(0.0, 0.0, 0.31), 0.0, 0.0, 1.0).is_some());
    }

    #[test]
    fn the_distance_text_rounds_and_switches_to_km() {
        let text = |d: f64| {
            let mut out = String::new();
            push_distance(&mut out, distance_key(d));
            out
        };
        assert_eq!(text(0.4), "0m");
        assert_eq!(text(123.4), "123m");
        assert_eq!(text(123.5), "124m");
        assert_eq!(text(9999.4), "9999m");
        assert_eq!(text(9999.5), "10.0km");
        assert_eq!(text(12_345.0), "12.3km");
        assert_eq!(text(12_360.0), "12.4km");
        assert_eq!(text(1_000_000.0), "1000.0km");
        assert_eq!(
            distance_key(123.2),
            distance_key(122.6),
            "same text, same key"
        );
    }

    #[test]
    fn the_text_is_rebuilt_only_when_something_shown_changes() {
        let mut font = uniform_font();
        let mut label = WaypointLabel::default();
        assert!(label.update(&mut font, "Base", 0xFF_AA00, Some(48.2)));
        assert_eq!(label.text(), "Base\u{a7}7 (48m)");
        assert!(!label.update(&mut font, "Base", 0xFF_AA00, Some(48.2)));
        assert!(
            !label.update(&mut font, "Base", 0xFF_AA00, Some(47.9)),
            "still 48m"
        );
        assert!(label.update(&mut font, "Base", 0xFF_AA00, Some(47.4)));
        assert_eq!(label.text(), "Base\u{a7}7 (47m)");
        assert!(
            label.update(&mut font, "Base", 0x55_FF55, Some(47.4)),
            "colour"
        );
        assert!(
            label.update(&mut font, "Mid", 0x55_FF55, Some(47.4)),
            "name"
        );
        assert!(
            label.update(&mut font, "Mid", 0x55_FF55, None),
            "distance hidden"
        );
        assert_eq!(label.text(), "Mid");
        assert!(!label.update(&mut font, "Mid", 0x55_FF55, None));
        label.invalidate();
        assert!(label.update(&mut font, "Mid", 0x55_FF55, None));
        assert_eq!(label.sheets(), [FontTexture::Ascii]);
    }

    #[test]
    fn emit_pushes_backdrop_marker_and_glyphs_through_walls() {
        let mut font = uniform_font();
        let mut label = WaypointLabel::default();
        label.update(&mut font, "Base", 0xFF_AA00, Some(48.0));
        // "Base" + " (48m)": the §7 code draws nothing; the space advances without a quad.
        let drawn = "Base(48m)".chars().count();
        let rel = DVec3::new(2.0, 1.0, 48.0);
        let matrix = label_matrix(rel, 0.0, 0.0, 1.0).unwrap();
        let mut recording = Recording::default();
        let mut scratch = LabelScratch::default();
        label.emit(matrix, &mut scratch, &mut recording);

        assert_eq!(recording.solid.len(), 1);
        let (pass, solid) = &recording.solid[0];
        assert_eq!(*pass, NameplatePass::ThroughWalls);
        assert_eq!(solid.len(), 12, "backdrop and marker, two triangles each");
        assert_eq!(solid[0].color, BACKDROP_RGBA);
        let marker = &solid[6..];
        let orange = argb_to_rgba(0xFFFF_AA00);
        assert!(
            marker.iter().all(|v| v.color == orange),
            "opaque waypoint colour"
        );
        // The marker is centred on the anchor (the proxy point on the anchor's ray).
        let centre = (Vec3::from(marker[0].pos) + Vec3::from(marker[2].pos)) / 2.0;
        assert!((centre - matrix.transform_point3(Vec3::ZERO)).length() < 1e-4);

        assert_eq!(recording.glyphs.len(), 1, "one sheet, one run");
        let (pass, texture, glyphs) = &recording.glyphs[0];
        assert_eq!(
            (*pass, *texture),
            (NameplatePass::ThroughWalls, FontTexture::Ascii)
        );
        assert_eq!(glyphs.len(), 6 * drawn);
        // The name in the waypoint colour, the distance in vanilla's §7 grey, all opaque.
        assert_eq!(glyphs[0].color, orange);
        assert_eq!(glyphs[6 * 4].color, argb_to_rgba(0xFFAA_AAAA));
        assert!(glyphs.iter().all(|v| v.color[3] == 1.0));

        // The label sits above the marker on screen.
        let camera = camera();
        let (_, backdrop_bottom, _) = project(&camera, matrix, 0.0, TEXT_Y + 8.0);
        let (_, marker_top, _) = project(&camera, matrix, 0.0, -MARKER_HALF);
        assert!(
            backdrop_bottom < marker_top,
            "{backdrop_bottom} < {marker_top}"
        );
    }
}
