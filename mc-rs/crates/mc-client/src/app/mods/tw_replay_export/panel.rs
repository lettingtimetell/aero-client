//! The export's progress panel, drawn on the CPU into an RGBA image.
//!
//! While an export renders, the frame graph works at the export's resolution offscreen, so the
//! window's picture is composed by `mc_render::export::PreviewCompositor`: the latest exported frame
//! scaled to fit, and this panel on top. It is small (about 560 x 132 px at 1080p) and redrawn a few
//! times a second, so a CPU rasteriser is plenty: a rounded dark panel, Segoe UI text (read from
//! `C:\Windows\Fonts` like the mod menu's) and a progress bar in the menu's accent blue. Without the
//! fonts the panel still shows the bar.

use std::sync::OnceLock;

use ab_glyph::{Font, FontRef, PxScale, ScaleFont};

/// Accent blue of the Aero mod menu.
const ACCENT: [u8; 3] = [64, 156, 255];

/// The two Segoe UI weights the panel uses, loaded once per process.
struct Fonts {
    regular: std::borrow::Cow<'static, [u8]>,
    semibold: std::borrow::Cow<'static, [u8]>,
}

fn fonts() -> Option<&'static Fonts> {
    static FONTS: OnceLock<Option<Fonts>> = OnceLock::new();
    FONTS
        .get_or_init(|| {
            let dir = std::env::var_os("WINDIR")
                .map(std::path::PathBuf::from)
                .unwrap_or_else(|| std::path::PathBuf::from("C:\\Windows"))
                .join("Fonts");
            // Mapped (the same pages the mod menu maps) when possible, read
            // into memory as before otherwise.
            let load = |name: &str| {
                let path = dir.join(name);
                mc_render::w21_mb::map_file_for_process(&path)
                    .map(std::borrow::Cow::Borrowed)
                    .or_else(|| std::fs::read(path).ok().map(std::borrow::Cow::Owned))
            };
            let regular = load("segoeui.ttf")?;
            let semibold = load("seguisb.ttf").unwrap_or_else(|| regular.clone());
            Some(Fonts { regular, semibold })
        })
        .as_ref()
}

/// What the panel says.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct PanelText {
    /// "Exporting video", "Finishing", ...
    pub title: String,
    /// "1920x1080 · 60 fps · H.264 (GPU)"
    pub detail: String,
    /// "1:23 left · frame 1,234 of 5,400 · Esc to cancel"
    pub status: String,
    /// 0..=1.
    pub fraction: f32,
}

/// An RGBA image (straight alpha).
pub(super) struct Canvas {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
}

impl Canvas {
    fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            pixels: vec![0; (width * height * 4) as usize],
        }
    }

    /// Blend `color` (straight alpha `alpha`, 0..1) over the pixel at (x, y).
    fn blend(&mut self, x: i32, y: i32, color: [u8; 3], alpha: f32) {
        if x < 0 || y < 0 || x >= self.width as i32 || y >= self.height as i32 || alpha <= 0.0 {
            return;
        }
        let o = ((y as u32 * self.width + x as u32) * 4) as usize;
        let dst_a = f32::from(self.pixels[o + 3]) / 255.0;
        let a = alpha.min(1.0);
        let out_a = a + dst_a * (1.0 - a);
        if out_a <= 0.0 {
            return;
        }
        for (dst, &src) in self.pixels[o..o + 3].iter_mut().zip(&color) {
            let value = (f32::from(src) * a + f32::from(*dst) * dst_a * (1.0 - a)) / out_a;
            *dst = value.round().clamp(0.0, 255.0) as u8;
        }
        self.pixels[o + 3] = (out_a * 255.0).round() as u8;
    }

    /// An anti-aliased rounded rectangle.
    #[allow(clippy::too_many_arguments)]
    fn rounded_rect(&mut self, x: f32, y: f32, w: f32, h: f32, r: f32, color: [u8; 3], alpha: f32) {
        let r = r.min(w * 0.5).min(h * 0.5);
        let (x0, y0) = (x.floor() as i32, y.floor() as i32);
        let (x1, y1) = ((x + w).ceil() as i32, (y + h).ceil() as i32);
        for py in y0..y1 {
            for px in x0..x1 {
                let (cx, cy) = (px as f32 + 0.5, py as f32 + 0.5);
                // Signed distance to the rounded box.
                let qx = (cx - (x + w * 0.5)).abs() - (w * 0.5 - r);
                let qy = (cy - (y + h * 0.5)).abs() - (h * 0.5 - r);
                let outside = (qx.max(0.0).powi(2) + qy.max(0.0).powi(2)).sqrt();
                let inside = qx.max(qy).min(0.0);
                let distance = outside + inside - r;
                let coverage = (0.5 - distance).clamp(0.0, 1.0);
                self.blend(px, py, color, alpha * coverage);
            }
        }
    }

    /// Draw `text` with its baseline at `baseline`, starting at `x` (or ending there, if
    /// `right_aligned`). Returns the advance width.
    #[allow(clippy::too_many_arguments)]
    fn text(
        &mut self,
        font: &[u8],
        text: &str,
        size: f32,
        x: f32,
        baseline: f32,
        color: [u8; 3],
        alpha: f32,
        right_aligned: bool,
    ) -> f32 {
        let Ok(font) = FontRef::try_from_slice(font) else {
            return 0.0;
        };
        let scale =
            PxScale::from(size * font.height_unscaled() / font.units_per_em().unwrap_or(1.0));
        let scaled = font.as_scaled(scale);
        let width = measure(&scaled, text);
        let mut pen = if right_aligned { x - width } else { x };
        let mut previous = None;
        for c in text.chars() {
            let id = scaled.glyph_id(c);
            if let Some(p) = previous {
                pen += scaled.kern(p, id);
            }
            let glyph = id.with_scale_and_position(scale, ab_glyph::point(pen, baseline));
            if let Some(outlined) = scaled.outline_glyph(glyph) {
                let bounds = outlined.px_bounds();
                outlined.draw(|gx, gy, coverage| {
                    self.blend(
                        bounds.min.x as i32 + gx as i32,
                        bounds.min.y as i32 + gy as i32,
                        color,
                        alpha * coverage,
                    );
                });
            }
            pen += scaled.h_advance(id);
            previous = Some(id);
        }
        width
    }
}

fn measure<F: Font, S: ScaleFont<F>>(scaled: &S, text: &str) -> f32 {
    let mut width = 0.0;
    let mut previous = None;
    for c in text.chars() {
        let id = scaled.glyph_id(c);
        if let Some(p) = previous {
            width += scaled.kern(p, id);
        }
        width += scaled.h_advance(id);
        previous = Some(id);
    }
    width
}

/// The panel's scale for a window `window_height` pixels tall: 1.0 at 1080p, never so small
/// that its text stops being readable in a small window.
pub(super) fn scale(window_height: u32) -> f32 {
    (window_height as f32 / 1080.0).clamp(0.9, 2.5)
}

/// The panel for a window of `window_height` pixels (it scales with the window, 1.0 at 1080p).
pub(super) fn draw(text: &PanelText, window_height: u32) -> Canvas {
    let k = scale(window_height);
    let (w, h) = ((560.0 * k).round() as u32, (132.0 * k).round() as u32);
    let mut canvas = Canvas::new(w, h);
    let (wf, hf) = (w as f32, h as f32);
    // Glass-like body with a faint rim.
    canvas.rounded_rect(0.0, 0.0, wf, hf, 16.0 * k, [255, 255, 255], 0.14);
    canvas.rounded_rect(
        1.0 * k,
        1.0 * k,
        wf - 2.0 * k,
        hf - 2.0 * k,
        15.0 * k,
        [14, 16, 22],
        0.86,
    );
    let pad = 22.0 * k;
    if let Some(fonts) = fonts() {
        canvas.text(
            &fonts.semibold,
            &text.title,
            21.0 * k,
            pad,
            pad + 20.0 * k,
            [255, 255, 255],
            1.0,
            false,
        );
        let percent = format!("{:.0}%", (text.fraction * 100.0).floor());
        canvas.text(
            &fonts.semibold,
            &percent,
            21.0 * k,
            wf - pad,
            pad + 20.0 * k,
            [255, 255, 255],
            1.0,
            true,
        );
        canvas.text(
            &fonts.regular,
            &text.detail,
            14.5 * k,
            pad,
            pad + 43.0 * k,
            [255, 255, 255],
            0.72,
            false,
        );
        canvas.text(
            &fonts.regular,
            &text.status,
            13.5 * k,
            pad,
            hf - pad + 2.0 * k,
            [255, 255, 255],
            0.62,
            false,
        );
    }
    // The bar.
    let (bar_y, bar_h) = (pad + 60.0 * k, 7.0 * k);
    let bar_w = wf - pad * 2.0;
    canvas.rounded_rect(pad, bar_y, bar_w, bar_h, bar_h * 0.5, [255, 255, 255], 0.14);
    let fill = (bar_w * text.fraction.clamp(0.0, 1.0)).max(bar_h);
    canvas.rounded_rect(pad, bar_y, fill, bar_h, bar_h * 0.5, ACCENT, 1.0);
    canvas
}

/// `1:05`, `12:40`, `1:02:03`.
pub(super) fn clock(seconds: f64) -> String {
    let total = seconds.max(0.0).round() as u64;
    let (h, m, s) = (total / 3600, (total / 60) % 60, total % 60);
    if h > 0 {
        format!("{h}:{m:02}:{s:02}")
    } else {
        format!("{m}:{s:02}")
    }
}

/// `1,234,567`.
pub(super) fn thousands(n: u64) -> String {
    let digits = n.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn number_formats() {
        assert_eq!(clock(0.0), "0:00");
        assert_eq!(clock(65.4), "1:05");
        assert_eq!(clock(3723.0), "1:02:03");
        assert_eq!(thousands(0), "0");
        assert_eq!(thousands(1234), "1,234");
        assert_eq!(thousands(1_234_567), "1,234,567");
    }

    #[test]
    fn panel_has_a_bar_whose_fill_follows_the_fraction() {
        let make = |fraction| {
            draw(
                &PanelText {
                    title: "Exporting video".into(),
                    detail: "1920x1080".into(),
                    status: "0:10 left".into(),
                    fraction,
                },
                1080,
            )
        };
        let half = make(0.5);
        assert_eq!((half.width, half.height), (560, 132));
        // Sample the bar's centre line at 25% and 75% of its width.
        let row = (22.0 + 60.0 + 3.5) as u32;
        let at = |c: &Canvas, x: u32| {
            let o = ((row * c.width + x) * 4) as usize;
            [c.pixels[o], c.pixels[o + 1], c.pixels[o + 2]]
        };
        let quarter = 22 + (516 / 4);
        let three_quarters = 22 + 516 * 3 / 4;
        assert!(at(&half, quarter)[2] > 200, "filled part is accent blue");
        assert!(
            at(&half, three_quarters)[2] < 150,
            "unfilled part is the dark track"
        );
        let full = make(1.0);
        assert!(at(&full, three_quarters)[2] > 200);
    }
}
