//! Render hooks for the mods: the Crosshair mod's custom crosshair and
//! the Shiny Pots mod's square potion glint.
//!
//! This crate draws; it does not read mod settings. The client pushes plain values in whenever
//! the settings change: the crosshair into
//! [`Hud::custom_crosshair`](crate::screens::hud::Hud::custom_crosshair) and the glint area
//! through [`set_shiny_pots`]. Both default to "off", which is vanilla's exact draw.

use std::cell::Cell;

use crate::screen_stack::ScreenCtx;
use crate::screens::container::render::{ENCHANTED_ITEM_GLINT_TEXTURE, GLINT_COLOR_ARGB};
use crate::widget::draw::{unpack_argb, UiBlend, UiCull, UiDepth};
use crate::widget::{DrawList, UiTexture};

/// The Crosshair mod's look. Sizes are GUI pixels, so it follows the GUI scale exactly like the
/// vanilla sprite.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CustomCrosshair {
    /// Length of each of the four arms; `0` draws no arms.
    pub length: i32,
    /// Space between the centre band and each arm.
    pub gap: i32,
    /// Width of the arms and side of the dot (at least 1).
    pub thickness: i32,
    /// Draw a `thickness`-sided square on the centre.
    pub dot: bool,
    /// `None`: white rectangles in the inverting blend, vanilla's look. `Some(argb)`: that
    /// colour, alpha blended.
    pub color: Option<u32>,
    /// Draw it while the camera is in third person.
    pub show_in_f5: bool,
}

impl CustomCrosshair {
    /// The rectangles to fill, `[left, top, right, bottom]` in GUI pixels, around the GUI pixel
    /// whose top-left corner is `(cx, cy)`; only the first `count` entries are used.
    ///
    /// Every part shares one band `thickness` pixels wide centred on that pixel; an even
    /// thickness puts its extra pixel right and down. Arms start `gap` pixels past the band and
    /// run `length` pixels outward, so with gap 0 they touch the band but the centre stays empty
    /// unless the dot is on. No two rectangles overlap, so the inverting blend never flips a
    /// pixel twice.
    #[must_use]
    pub fn rects(&self, cx: i32, cy: i32) -> ([[i32; 4]; 5], usize) {
        let thickness = self.thickness.max(1);
        let gap = self.gap.max(0);
        let length = self.length.max(0);
        let (lo_x, lo_y) = (cx - (thickness - 1) / 2, cy - (thickness - 1) / 2);
        let (hi_x, hi_y) = (lo_x + thickness, lo_y + thickness);
        let mut out = [[0; 4]; 5];
        let mut count = 0;
        if length > 0 {
            for rect in [
                [lo_x - gap - length, lo_y, lo_x - gap, hi_y],
                [hi_x + gap, lo_y, hi_x + gap + length, hi_y],
                [lo_x, lo_y - gap - length, hi_x, lo_y - gap],
                [lo_x, hi_y + gap, hi_x, hi_y + gap + length],
            ] {
                out[count] = rect;
                count += 1;
            }
        }
        if self.dot {
            out[count] = [lo_x, lo_y, hi_x, hi_y];
            count += 1;
        }
        (out, count)
    }

    /// Draw it on the screen's centre pixel, where the vanilla sprite's centre texel lands
    /// (`GuiIngame` blits the 16x16 cell at `(w/2 - 7, h/2 - 7)`).
    pub fn draw(&self, d: &mut DrawList, ctx: &ScreenCtx) {
        let (rects, count) = self.rects(ctx.width / 2, ctx.height / 2);
        for &[left, top, right, bottom] in &rects[..count] {
            match self.color {
                None => d.rect_invert(left, top, right, bottom),
                Some(argb) => d.rect(left, top, right, bottom, argb),
            }
        }
    }
}

/// Which square the Shiny Pots mod fills with glint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShinyPotsArea {
    /// The 16x16 item icon.
    ItemBox,
    /// 1.7's 20x20 square, two pixels past the icon on every side.
    WholeSlot,
}

thread_local! {
    /// The Shiny Pots setting, per thread so parallel tests cannot see each other's value. The
    /// client sets it on the thread that builds the UI.
    static SHINY_POTS: Cell<Option<ShinyPotsArea>> = const { Cell::new(None) };
}

/// Turn the Shiny Pots mod's square glint on (`Some`) or off (`None`, vanilla) for GUI item
/// draws on this thread.
pub fn set_shiny_pots(area: Option<ShinyPotsArea>) {
    SHINY_POTS.with(|cell| cell.set(area));
}

/// The current Shiny Pots setting on this thread.
#[must_use]
pub fn shiny_pots() -> Option<ShinyPotsArea> {
    SHINY_POTS.with(Cell::get)
}

/// The two layers of 1.7's square glint: `(period in ms, skew)`. Layer `j` scrolls with period
/// `3000 + 1873 j` and leans by `4` (`j = 0`) or `-1` (`j = 1`) texels per pixel down.
const SQUARE_GLINT_LAYERS: [(u64, f32); 2] = [(3_000, 4.0), (4_873, -1.0)];

/// One square-glint texture coordinate: `(s + dx + skew * dy) / 256, dy / 256`, where `s` is
/// the layer's scroll (`0..256`) and `(dx, dy)` the GUI-pixel offset into the square.
#[must_use]
pub fn square_glint_uv(time_ms: u64, layer: usize, dx: f32, dy: f32) -> [f32; 2] {
    let (period, skew) = SQUARE_GLINT_LAYERS[layer];
    let scroll = (time_ms % period) as f32 / period as f32 * 256.0;
    [(scroll + dx + skew * dy) / 256.0, dy / 256.0]
}

/// Draw the Shiny Pots glint over a GUI item at `(x, y)`: two scrolling layers of the
/// enchantment glint across the whole square, in the foil's colour and `SRC_COLOR, ONE` blend.
///
/// The model foil is drawn with `GL_EQUAL` against the item's own depth, which is what clips it
/// to the bottle. This square has nothing to match outside the bottle, so it goes through the
/// no-depth-test foil pipeline instead.
pub fn draw_square_glint(draw: &mut DrawList, x: i32, y: i32, area: ShinyPotsArea, time_ms: u64) {
    static TEXTURE: std::sync::LazyLock<UiTexture> =
        std::sync::LazyLock::new(|| UiTexture::gui(ENCHANTED_ITEM_GLINT_TEXTURE));
    let (left, top, size) = match area {
        ShinyPotsArea::ItemBox => (x, y, 16),
        ShinyPotsArea::WholeSlot => (x - 2, y - 2, 20),
    };
    let (l, t, n) = (left as f32, top as f32, size as f32);
    let (r, b) = (l + n, t + n);
    let color = [unpack_argb(GLINT_COLOR_ARGB); 3];
    draw.with_render_state(UiDepth::Overlay, UiCull::Back, |draw| {
        for layer in 0..SQUARE_GLINT_LAYERS.len() {
            let uv = |dx: f32, dy: f32| square_glint_uv(time_ms, layer, dx, dy);
            // The same corner order (and so winding) as `DrawList::quad_rect`.
            draw.triangle_with_blend(
                &TEXTURE,
                UiBlend::Glint,
                [[l, t], [l, b], [r, t]],
                [uv(0.0, 0.0), uv(0.0, n), uv(n, 0.0)],
                color,
            );
            draw.triangle_with_blend(
                &TEXTURE,
                UiBlend::Glint,
                [[l, b], [r, b], [r, t]],
                [uv(0.0, n), uv(n, n), uv(n, 0.0)],
                color,
            );
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::widget::ScaledResolution;

    fn crosshair() -> CustomCrosshair {
        CustomCrosshair {
            length: 4,
            gap: 2,
            thickness: 1,
            dot: false,
            color: None,
            show_in_f5: true,
        }
    }

    #[test]
    fn a_one_pixel_cross_is_four_arms_around_an_empty_centre() {
        let (rects, count) = crosshair().rects(100, 50);
        assert_eq!(count, 4);
        assert_eq!(
            &rects[..count],
            &[
                [94, 50, 98, 51],
                [103, 50, 107, 51],
                [100, 44, 101, 48],
                [100, 53, 101, 57],
            ]
        );
    }

    #[test]
    fn an_even_thickness_puts_its_extra_pixel_right_and_down() {
        let style = CustomCrosshair {
            thickness: 2,
            gap: 0,
            dot: true,
            ..crosshair()
        };
        let (rects, count) = style.rects(10, 10);
        assert_eq!(count, 5);
        assert_eq!(rects[4], [10, 10, 12, 12], "the dot is the band square");
        assert_eq!(
            rects[0],
            [6, 10, 10, 12],
            "gap 0: the left arm touches the band"
        );
        let odd = CustomCrosshair {
            thickness: 3,
            dot: true,
            length: 0,
            ..crosshair()
        };
        let (rects, count) = odd.rects(10, 10);
        assert_eq!((count, rects[0]), (1, [9, 9, 12, 12]), "odd: centred");
    }

    #[test]
    fn no_rectangles_overlap() {
        for thickness in 1..=4 {
            for gap in 0..=3 {
                let style = CustomCrosshair {
                    thickness,
                    gap,
                    dot: true,
                    ..crosshair()
                };
                let (rects, count) = style.rects(0, 0);
                for (i, a) in rects[..count].iter().enumerate() {
                    for b in &rects[i + 1..count] {
                        let disjoint = a[2] <= b[0] || b[2] <= a[0] || a[3] <= b[1] || b[3] <= a[1];
                        assert!(disjoint, "{a:?} overlaps {b:?} (t={thickness}, gap={gap})");
                    }
                }
            }
        }
    }

    #[test]
    fn inverted_mode_draws_white_invert_quads_and_custom_mode_its_colour() {
        let ctx = ScreenCtx::new(ScaledResolution::new(640, 480, 1, false), 0.0);
        let mut d = DrawList::new();
        crosshair().draw(&mut d, &ctx);
        assert_eq!(d.vertices().len(), 4 * 6);
        assert!(d.batches().iter().all(|b| b.blend == UiBlend::Invert));
        assert!(d.vertices().iter().all(|v| v.color == [1.0; 4]));

        let mut d = DrawList::new();
        let custom = CustomCrosshair {
            color: Some(0x80FF_0000),
            ..crosshair()
        };
        custom.draw(&mut d, &ctx);
        assert!(d.batches().iter().all(|b| b.blend == UiBlend::Alpha));
        assert!(d
            .vertices()
            .iter()
            .all(|v| v.color == unpack_argb(0x80FF_0000)));
    }

    #[test]
    fn the_shiny_pots_setting_is_per_thread_and_off_by_default() {
        assert_eq!(shiny_pots(), None);
        set_shiny_pots(Some(ShinyPotsArea::WholeSlot));
        assert_eq!(shiny_pots(), Some(ShinyPotsArea::WholeSlot));
        std::thread::spawn(|| assert_eq!(shiny_pots(), None))
            .join()
            .unwrap();
        set_shiny_pots(None);
        assert_eq!(shiny_pots(), None);
    }

    #[test]
    fn the_square_glint_covers_the_chosen_square_in_two_overlay_foil_layers() {
        for (area, (l, t, r, b)) in [
            (ShinyPotsArea::ItemBox, (40.0, 60.0, 56.0, 76.0)),
            (ShinyPotsArea::WholeSlot, (38.0, 58.0, 58.0, 78.0)),
        ] {
            let mut d = DrawList::new();
            draw_square_glint(&mut d, 40, 60, area, 1_234);
            let vertices = d.vertices();
            assert_eq!(vertices.len(), 12, "two layers of two triangles");
            for v in vertices {
                assert!(v.pos[0] == l || v.pos[0] == r, "{:?}", v.pos);
                assert!(v.pos[1] == t || v.pos[1] == b, "{:?}", v.pos);
                assert_eq!(v.color, unpack_argb(GLINT_COLOR_ARGB));
            }
            for batch in d.batches() {
                assert_eq!(batch.blend, UiBlend::Glint);
                assert_eq!(batch.depth, UiDepth::Overlay, "not the GL_EQUAL model foil");
                assert_eq!(batch.texture, UiTexture::gui(ENCHANTED_ITEM_GLINT_TEXTURE));
            }
            // The state window is restored for whatever the caller draws next.
            d.rect(0, 0, 1, 1, 0xFFFF_FFFF);
            assert_eq!(d.batches().last().unwrap().depth, UiDepth::Overlay);
        }
    }

    #[test]
    fn square_glint_uvs_follow_the_1_7_layers() {
        // Layer 0 at t = 1500 ms is half way through its 3000 ms period: scroll 128.
        assert_eq!(square_glint_uv(1_500, 0, 0.0, 0.0), [0.5, 0.0]);
        assert_eq!(
            square_glint_uv(1_500, 0, 16.0, 16.0),
            [(128.0 + 16.0 + 64.0) / 256.0, 0.0625]
        );
        // Layer 1 leans the other way and wraps on its own 4873 ms period.
        assert_eq!(
            square_glint_uv(4_873, 1, 0.0, 16.0),
            [-16.0 / 256.0, 0.0625]
        );
    }
}
