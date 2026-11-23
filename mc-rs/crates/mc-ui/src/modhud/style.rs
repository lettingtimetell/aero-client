//! The shared look of text readouts: Minecraft font, optional drop shadow, optional translucent
//! box.

use mc_mods::{presets, ModSettings, OptionValue};

use crate::font::FontRenderer;
use crate::widget::DrawList;

/// Horizontal padding inside the background box, GUI pixels.
pub const BOX_PAD_X: f32 = 4.0;
/// Height of the background box. The 8-pixel glyph cell sits 4 pixels from the top.
pub const BOX_HEIGHT: f32 = 16.0;
/// Text baseline offset inside the box.
pub const BOX_TEXT_Y: f32 = 4.0;

/// How one readout is drawn. Build it with [`HudTextStyle::read`] in your element's `sync`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HudTextStyle {
    /// `0xAARRGGBB` text colour.
    pub color: u32,
    /// Vanilla's one-pixel drop shadow.
    pub shadow: bool,
    /// The box colour, or `None` for no box.
    pub background: Option<u32>,
    /// Wrap the text in `[` `]`.
    pub brackets: bool,
}

impl Default for HudTextStyle {
    fn default() -> Self {
        HudTextStyle {
            color: 0xFFFF_FFFF,
            shadow: true,
            background: Some(0x6F00_0000),
            brackets: false,
        }
    }
}

impl HudTextStyle {
    /// Read the [`presets`] options of `mod_id`. Options the mod does not have keep this type's
    /// defaults, except that a mod without a `background` toggle gets no box.
    #[must_use]
    pub fn read(settings: &ModSettings, mod_id: &str) -> HudTextStyle {
        let defaults = HudTextStyle::default();
        let flag = |id: &str, fallback: bool| match settings.value(mod_id, id) {
            Some(OptionValue::Bool(on)) => *on,
            _ => fallback,
        };
        let color = |id: &str, fallback: u32| match settings.value(mod_id, id) {
            Some(OptionValue::Color(argb)) => *argb,
            _ => fallback,
        };
        let background = flag(presets::BACKGROUND_ID, false).then(|| {
            color(
                presets::BACKGROUND_COLOR_ID,
                defaults.background.unwrap_or(0x6F00_0000),
            )
        });
        HudTextStyle {
            color: color(presets::TEXT_COLOR_ID, defaults.color),
            shadow: flag(presets::TEXT_SHADOW_ID, defaults.shadow),
            background,
            brackets: flag(presets::BRACKETS_ID, false),
        }
    }

    /// `text` with brackets applied when the style asks for them.
    pub fn decorate(&self, text: &str, out: &mut String) {
        out.clear();
        if self.brackets {
            out.push('[');
            out.push_str(text);
            out.push(']');
        } else {
            out.push_str(text);
        }
    }

    /// The size [`HudTextStyle::draw`] covers for `text` (already decorated).
    #[must_use]
    pub fn size(&self, font: &FontRenderer, text: &str) -> (f32, f32) {
        let width = font.string_width(text).max(0) as f32;
        if self.background.is_some() {
            (width + BOX_PAD_X * 2.0 - 1.0, BOX_HEIGHT)
        } else {
            // The font's advance already includes one trailing pixel; the shadow adds one more.
            (width, font.font_height() as f32)
        }
    }

    /// Draw `text` at local `(x, y)`, its top-left corner (box included), sized as
    /// [`HudTextStyle::size`].
    pub fn draw(&self, d: &mut DrawList, font: &mut FontRenderer, text: &str, x: f32, y: f32) {
        let (w, h) = self.size(font, text);
        let (text_x, text_y) = match self.background {
            Some(argb) => {
                d.quad_rect(
                    &crate::widget::UiTexture::Solid,
                    (x, y, x + w, y + h),
                    (0.0, 0.0, 1.0, 1.0),
                    argb,
                );
                (x + BOX_PAD_X, y + BOX_TEXT_Y)
            }
            None => (x, y),
        };
        font.draw_string(d, text, text_x, text_y, self.color as i32, self.shadow);
    }

    /// Draw `text` centred horizontally in a box `width` wide (for readouts whose box should not
    /// jitter as the number changes width).
    pub fn draw_centered(
        &self,
        d: &mut DrawList,
        font: &mut FontRenderer,
        text: &str,
        x: f32,
        y: f32,
        width: f32,
    ) {
        let text_width = font.string_width(text).max(0) as f32;
        let text_y = match self.background {
            Some(argb) => {
                d.quad_rect(
                    &crate::widget::UiTexture::Solid,
                    (x, y, x + width, y + BOX_HEIGHT),
                    (0.0, 0.0, 1.0, 1.0),
                    argb,
                );
                y + BOX_TEXT_Y
            }
            None => y,
        };
        let text_x = (x + (width - text_width + 1.0) * 0.5).round();
        font.draw_string(d, text, text_x, text_y, self.color as i32, self.shadow);
    }
}
