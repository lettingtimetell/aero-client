//! The recorded mouse pointer, drawn over the screens a replay shows,
//! shaped like the desktop's standard arrow pointer.
//!
//! The arrow is this module's own outline ([`OUTLINE`]: a polygon in pixels at 100% display
//! scaling, its tip at the origin), rasterised here: each pixel's coverage of the outline, and of
//! the white body one border width inside it, is measured on a grid of [`SUBSAMPLES`]² points.
//! That gives a white arrow with a thin black border and smooth (antialiased) edges, laid over a
//! faint soft shadow down and to the right.
//!
//! It is drawn at the target's own pixel resolution, not in GUI pixels: the rasterised pixels are
//! kept as a private draw list of one-pixel quads (each run of one colour in a row is one quad),
//! built once per size and appended every frame a recorded screen is shown, through a transform
//! that puts its tip on the recorded position and each of its pixels on one pixel of the target.
//! Its size is the desktop pointer's at the window's display scaling
//! (`ShellApp::window_scale_changed`); in a video export, the same share of the frame's height as
//! at 1080p.
//!
//! Cost: nothing unless a recorded screen is shown; then one transform and a copy of the cached
//! quads a frame (about 200 at 100% display scaling, 650 at 200%; the inventory screen under it
//! draws far more), with no allocation once the frame's list has grown.

use mc_ui::widget::{DrawList, Transform, UiTexture};

/// The arrow's outline at 100% display scaling, in pixels, from the tip (the hotspot) down the
/// straight left edge, up to the notch, down the tail and across its end, up to the shoulder and
/// out along it, then back up the 45° edge to the tip. About 12 by 19 pixels.
const OUTLINE: [[f32; 2]; 7] = [
    [0.0, 0.0],
    [0.0, 16.8],
    [4.0, 13.0],
    [6.6, 19.0],
    [10.3, 17.4],
    [7.7, 11.6],
    [11.6, 11.6],
];
/// The black border's width at 100%, in pixels.
const BORDER: f32 = 1.0;
/// Coverage samples per pixel side.
const SUBSAMPLES: usize = 8;
/// The shadow's offset down and to the right, and its blur radius, at 100%, in pixels.
const SHADOW_OFFSET: f32 = 1.0;
const SHADOW_BLUR: f32 = 1.0;
/// The shadow's strongest opacity.
const SHADOW_ALPHA: f32 = 0.3;
/// The sizes drawn: a display scaling outside this range is clamped.
const SIZE_RANGE: (f32, f32) = (0.5, 4.0);
/// Sizes are drawn in steps of a quarter, so the cache is rebuilt only for a real change.
const SIZE_STEPS: f32 = 4.0;
/// In an export, the pointer is as tall a share of the frame as at 1080p and 100%.
const EXPORT_REFERENCE_HEIGHT: f32 = 1080.0;

/// One rasterised pointer, origin at the tip.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Raster {
    pub(crate) width: usize,
    pub(crate) height: usize,
    /// Row-major straight-alpha RGBA (always grey: `r == g == b`).
    pub(crate) rgba: Vec<[u8; 4]>,
    /// The arrow's own opacity per pixel, without the shadow.
    pub(crate) arrow_alpha: Vec<u8>,
}

/// Whether `p` is inside `polygon` (even-odd rule).
fn inside(polygon: &[[f32; 2]], [px, py]: [f32; 2]) -> bool {
    let mut inside = false;
    let mut j = polygon.len() - 1;
    for i in 0..polygon.len() {
        let ([xi, yi], [xj, yj]) = (polygon[i], polygon[j]);
        if (yi > py) != (yj > py) && px < (xj - xi) * (py - yi) / (yj - yi) + xi {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// The distance from `p` to the nearest edge of `polygon`.
fn edge_distance(polygon: &[[f32; 2]], [px, py]: [f32; 2]) -> f32 {
    let mut nearest = f32::MAX;
    let mut j = polygon.len() - 1;
    for i in 0..polygon.len() {
        let ([ax, ay], [bx, by]) = (polygon[j], polygon[i]);
        let (dx, dy) = (bx - ax, by - ay);
        let t = (((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)).clamp(0.0, 1.0);
        let (qx, qy) = (ax + t * dx, ay + t * dy);
        nearest = nearest.min(((px - qx) * (px - qx) + (py - qy) * (py - qy)).sqrt());
        j = i;
    }
    nearest
}

/// A box blur of radius `radius` (pixels outside the grid count as empty).
fn box_blur(source: &[f32], width: usize, height: usize, radius: usize) -> Vec<f32> {
    let side = 2 * radius + 1;
    let count = (side * side) as f32;
    let mut out = vec![0.0; source.len()];
    for y in 0..height {
        for x in 0..width {
            let mut sum = 0.0;
            for yy in y.saturating_sub(radius)..(y + radius + 1).min(height) {
                for xx in x.saturating_sub(radius)..(x + radius + 1).min(width) {
                    sum += source[yy * width + xx];
                }
            }
            out[y * width + x] = sum / count;
        }
    }
    out
}

/// The pointer at `size` times its 100% size.
pub(crate) fn rasterize(size: f32) -> Raster {
    let size = size.clamp(SIZE_RANGE.0, SIZE_RANGE.1);
    let outline = OUTLINE.map(|[x, y]| [x * size, y * size]);
    let border = BORDER * size;
    let (right, bottom) = outline
        .iter()
        .fold((0.0f32, 0.0f32), |(r, b), [x, y]| (r.max(*x), b.max(*y)));
    let offset = (SHADOW_OFFSET * size).round().max(1.0) as usize;
    let blur = (SHADOW_BLUR * size).round().max(1.0) as usize;
    let width = right.ceil() as usize + offset + 2 * blur + 1;
    let height = bottom.ceil() as usize + offset + 2 * blur + 1;
    // Coverage of the whole arrow and of its white body.
    let mut outer = vec![0.0f32; width * height];
    let mut body = vec![0.0f32; width * height];
    let step = 1.0 / SUBSAMPLES as f32;
    let samples = (SUBSAMPLES * SUBSAMPLES) as f32;
    for y in 0..height.min(bottom.ceil() as usize + 1) {
        for x in 0..width.min(right.ceil() as usize + 1) {
            let (mut whole, mut white) = (0u32, 0u32);
            for sy in 0..SUBSAMPLES {
                for sx in 0..SUBSAMPLES {
                    let p = [
                        x as f32 + (sx as f32 + 0.5) * step,
                        y as f32 + (sy as f32 + 0.5) * step,
                    ];
                    if inside(&outline, p) {
                        whole += 1;
                        if edge_distance(&outline, p) >= border {
                            white += 1;
                        }
                    }
                }
            }
            outer[y * width + x] = whole as f32 / samples;
            body[y * width + x] = white as f32 / samples;
        }
    }
    // The shadow: the arrow's coverage moved down and right, softened by two box blurs.
    let mut shadow = vec![0.0f32; width * height];
    for y in offset..height {
        for x in offset..width {
            shadow[y * width + x] = outer[(y - offset) * width + (x - offset)];
        }
    }
    let shadow = box_blur(&box_blur(&shadow, width, height, blur), width, height, blur);
    // The arrow (white body, black border) over the shadow, straight alpha.
    let to_byte = |value: f32| (value.clamp(0.0, 1.0) * 255.0).round() as u8;
    let mut rgba = Vec::with_capacity(width * height);
    let mut arrow_alpha = Vec::with_capacity(width * height);
    for ((&arrow, &white), &shade) in outer.iter().zip(&body).zip(&shadow) {
        let alpha = arrow + SHADOW_ALPHA * shade * (1.0 - arrow);
        let grey = to_byte(if alpha > 0.0 { white / alpha } else { 0.0 });
        rgba.push([grey, grey, grey, to_byte(alpha)]);
        arrow_alpha.push(to_byte(arrow));
    }
    Raster {
        width,
        height,
        rgba,
        arrow_alpha,
    }
}

/// `raster` as one-pixel quads (each run of one colour in a row is one quad), in its pixels.
fn quads(raster: &Raster) -> DrawList {
    let mut list = DrawList::new();
    for (y, row) in raster.rgba.chunks_exact(raster.width).enumerate() {
        let mut x = 0;
        while x < row.len() {
            let pixel = row[x];
            let mut end = x + 1;
            while end < row.len() && row[end] == pixel {
                end += 1;
            }
            if pixel[3] != 0 {
                let [r, g, b, a] = pixel.map(u32::from);
                list.quad_rect(
                    &UiTexture::Solid,
                    (x as f32, y as f32, end as f32, y as f32 + 1.0),
                    (0.0, 0.0, 1.0, 1.0),
                    (a << 24) | (r << 16) | (g << 8) | b,
                );
            }
            x = end;
        }
    }
    list
}

/// The pointer's size in a frame of `export_height` pixels of a video export.
pub(crate) fn export_size(export_height: u32) -> f32 {
    export_height as f32 / EXPORT_REFERENCE_HEIGHT
}

/// The recorded pointer, rasterised once per size (`ReplayPlayState::pointer`).
#[derive(Debug)]
pub(crate) struct PointerCache {
    /// The window's display scaling (1.0 at 100%), from `ShellApp::window_scale_changed`.
    window_scale: f32,
    /// The size [`Self::quads`] holds, in [`SIZE_STEPS`] steps (`None`: not built yet).
    built: Option<u32>,
    quads: DrawList,
    /// Test pictures only: draw the pointer replays drew before this one.
    #[cfg(test)]
    pub(crate) legacy: bool,
}

impl Default for PointerCache {
    fn default() -> Self {
        Self {
            window_scale: 1.0,
            built: None,
            quads: DrawList::new(),
            #[cfg(test)]
            legacy: false,
        }
    }
}

impl PointerCache {
    /// The window moved to a display with another scaling (or opened on one).
    pub(crate) fn set_window_scale(&mut self, scale_factor: f64) {
        if scale_factor.is_finite() && scale_factor > 0.0 {
            self.window_scale = scale_factor as f32;
        }
    }

    /// The window's display scaling: the size the desktop's own pointer has in it.
    pub(crate) fn window_scale(&self) -> f32 {
        self.window_scale
    }

    /// Draw the pointer with its tip at `tip` (GUI pixels of a target with `gui_scale` of its
    /// pixels per GUI pixel), at `size` times its 100% size.
    pub(crate) fn draw(&mut self, list: &mut DrawList, tip: (i32, i32), gui_scale: i32, size: f32) {
        #[cfg(test)]
        if self.legacy {
            draw_legacy(list, tip);
            return;
        }
        let steps = (size.clamp(SIZE_RANGE.0, SIZE_RANGE.1) * SIZE_STEPS).round() as u32;
        if self.built != Some(steps) {
            self.quads = quads(&rasterize(steps as f32 / SIZE_STEPS));
            self.built = Some(steps);
        }
        let pixel = 1.0 / gui_scale.max(1) as f32;
        list.push_transform(Transform::translate(tip.0 as f32, tip.1 as f32).then_scale(pixel));
        list.append(&self.quads);
        list.pop_transform();
    }

    /// The cached quads, for tests.
    #[cfg(test)]
    pub(crate) fn cached(&self) -> &DrawList {
        &self.quads
    }
}

/// The pointer replays drew before this module: an arrow 7 GUI pixels wide and 10 tall with a
/// black outline. Kept for the before/after pictures.
#[cfg(test)]
fn draw_legacy(list: &mut DrawList, (x, y): (i32, i32)) {
    const ROWS: [&str; 10] = [
        "B", "BB", "BWB", "BWWB", "BWWWB", "BWWWWB", "BWWWWWB", "BWWBBBB", "BWB", "BB",
    ];
    for (row, pixels) in ROWS.iter().enumerate() {
        let top = y + row as i32;
        let mut run_start = 0usize;
        let bytes = pixels.as_bytes();
        while run_start < bytes.len() {
            let colour = bytes[run_start];
            let mut run_end = run_start + 1;
            while run_end < bytes.len() && bytes[run_end] == colour {
                run_end += 1;
            }
            let argb = if colour == b'W' {
                0xFFFF_FFFF
            } else {
                0xFF00_0000
            };
            list.rect(x + run_start as i32, top, x + run_end as i32, top + 1, argb);
            run_start = run_end;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn alpha_at(raster: &Raster, x: usize, y: usize) -> u8 {
        raster.rgba[y * raster.width + x][3]
    }

    /// The bounding box `(width, height)` of the pixels at least half covered by the arrow.
    fn arrow_box(raster: &Raster) -> (usize, usize) {
        let (mut right, mut bottom) = (0, 0);
        for y in 0..raster.height {
            for x in 0..raster.width {
                if raster.arrow_alpha[y * raster.width + x] >= 128 {
                    right = right.max(x + 1);
                    bottom = bottom.max(y + 1);
                }
            }
        }
        (right, bottom)
    }

    #[test]
    fn the_arrow_is_white_with_a_thin_black_border_and_smooth_edges() {
        for size in [1.0, 1.5, 2.0] {
            let raster = rasterize(size);
            let opaque: Vec<[u8; 4]> = raster
                .rgba
                .iter()
                .copied()
                .filter(|pixel| pixel[3] == 255)
                .collect();
            let white = opaque.iter().filter(|pixel| pixel[0] == 255).count();
            let black = opaque.iter().filter(|pixel| pixel[0] == 0).count();
            let soft = raster
                .arrow_alpha
                .iter()
                .filter(|alpha| (1..255).contains(*alpha))
                .count();
            assert!(
                raster.rgba.iter().all(|[r, g, b, _]| r == g && g == b),
                "{size}: grey only"
            );
            // At 100%: 37 pure white pixels inside a border with 14 pure black ones, and 41
            // partly covered edge pixels; more of each at larger sizes.
            assert!(
                white as f32 >= 30.0 * size * size,
                "{size}: {white} white pixels"
            );
            assert!(black as f32 >= 12.0 * size, "{size}: {black} black pixels");
            assert!(
                soft as f32 >= 20.0 * size,
                "{size}: {soft} antialiased edge pixels"
            );
            // The straight left edge is solid black below the tip (the 45° edge meets it there).
            for y in 2..(15.0 * size) as usize {
                let pixel = raster.rgba[y * raster.width];
                assert_eq!(pixel, [0, 0, 0, 255], "{size}: left edge at row {y}");
            }
            // Inside the border the body is pure white, e.g. a few pixels right of the edge.
            let x = (3.0 * size) as usize;
            let y = (8.0 * size) as usize;
            assert_eq!(raster.rgba[y * raster.width + x], [255, 255, 255, 255]);
        }
    }

    #[test]
    fn the_tip_is_the_hotspot() {
        for size in [0.5, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0] {
            let raster = rasterize(size);
            // The arrow starts in the tip's own pixel: dark and well covered (the 45° edge
            // halves it), and the pixel just below is solid.
            let tip = raster.rgba[0];
            assert!(tip[0] < 64 && tip[3] >= 100, "{size}: tip pixel {tip:?}");
            assert!(
                raster.arrow_alpha[raster.width] >= 200,
                "{size}: below the tip"
            );
            // Nothing of the arrow is left of or above it (the raster starts at the tip), and
            // the first row holds only the tip's own pixel and its edge.
            let first_row = (0..raster.width)
                .filter(|x| raster.arrow_alpha[*x] > 0)
                .count();
            assert!(
                first_row <= 2,
                "{size}: {first_row} pixels in the tip's row"
            );
        }
    }

    #[test]
    fn its_size_is_the_desktop_pointers_at_the_display_scaling() {
        // 11 by 19 pixels at 100% (the shoulder's corner pixel is less than half covered).
        assert_eq!(arrow_box(&rasterize(1.0)), (11, 19));
        assert_eq!(arrow_box(&rasterize(1.5)), (16, 28));
        assert_eq!(arrow_box(&rasterize(2.0)), (22, 38));
        // Out of range: clamped.
        assert_eq!(rasterize(9.0), rasterize(4.0));
        assert_eq!(rasterize(0.1), rasterize(0.5));
        assert_eq!(export_size(1080), 1.0);
        assert_eq!(export_size(2160), 2.0);
    }

    #[test]
    fn the_shadow_is_faint_soft_and_down_right() {
        let raster = rasterize(1.0);
        let mut shadow_pixels = 0;
        for y in 0..raster.height {
            for x in 0..raster.width {
                let i = y * raster.width + x;
                if raster.arrow_alpha[i] == 0 && raster.rgba[i][3] > 0 {
                    shadow_pixels += 1;
                    assert_eq!(raster.rgba[i][0], 0, "the shadow is black");
                    assert!(raster.rgba[i][3] <= (SHADOW_ALPHA * 255.0) as u8 + 1);
                }
            }
        }
        assert!(shadow_pixels > 20, "{shadow_pixels} shadow pixels");
        // Right of the tip, above the 45° edge, there is none (it falls down and right).
        assert_eq!(alpha_at(&raster, 4, 0), 0);
    }

    #[test]
    fn it_is_built_once_and_drawn_at_the_targets_own_pixels() {
        let mut cache = PointerCache::default();
        let mut list = DrawList::new();
        cache.draw(&mut list, (100, 50), 3, 1.0);
        let built = cache.cached().vertices().len();
        let buffer = cache.cached().vertices().as_ptr();
        eprintln!("the pointer at 100%: {} quads", built / 6);
        assert!(built > 0);
        assert_eq!(list.vertices().len(), built);
        assert_eq!(list.transform_depth(), 0, "the transform is popped");
        // The tip is at the recorded GUI position, and one raster pixel is one target pixel:
        // a third of a GUI pixel at GUI scale 3.
        let (mut left, mut top, mut right) = (f32::MAX, f32::MAX, f32::MIN);
        for vertex in list.vertices() {
            left = left.min(vertex.pos[0]);
            top = top.min(vertex.pos[1]);
            right = right.max(vertex.pos[0]);
        }
        assert_eq!((left, top), (100.0, 50.0));
        let raster = rasterize(1.0);
        let widest = (0..raster.height)
            .filter_map(|y| {
                (0..raster.width)
                    .rev()
                    .find(|x| raster.rgba[y * raster.width + x][3] > 0)
            })
            .max()
            .expect("pixels");
        assert!((right - (100.0 + (widest + 1) as f32 / 3.0)).abs() < 1e-4);
        // Same size again: nothing rebuilt; another size: rebuilt once.
        cache.draw(&mut list, (0, 0), 2, 1.0);
        assert_eq!(cache.cached().vertices().as_ptr(), buffer);
        assert_eq!(list.vertices().len(), 2 * built);
        cache.draw(&mut list, (0, 0), 2, 2.0);
        assert!(cache.cached().vertices().len() > built);
    }

    #[test]
    fn the_window_scale_is_kept_when_sane() {
        let mut cache = PointerCache::default();
        assert_eq!(cache.window_scale(), 1.0);
        cache.set_window_scale(1.5);
        assert_eq!(cache.window_scale(), 1.5);
        cache.set_window_scale(f64::NAN);
        cache.set_window_scale(0.0);
        assert_eq!(cache.window_scale(), 1.5);
    }
}
