//! Shared furniture every screen uses: backgrounds, hovering text, and the unicode-font rule.
//!
//! java: `client/gui/GuiScreen.java` (`drawDefaultBackground`, `drawBackground`,
//! `drawWorldBackground`, `drawHoveringText`)
//!
//! Everything here is arithmetic over a [`DrawList`], so a screen's whole background can be
//! asserted on without a GPU.

use mc_assets::Language;
use mc_render::settings::GameSettings;

use crate::font::FontRenderer;
use crate::widget::draw::{DrawList, Rect, UiTexture, OPTIONS_BACKGROUND};

/// `textures/gui/title/minecraft.png` — the wordmark on the main menu.
pub const TITLE_TEXTURE: &str = "textures/gui/title/minecraft.png";
/// `textures/misc/unknown_server.png` — the placeholder server icon.
pub const UNKNOWN_SERVER_TEXTURE: &str = "textures/misc/unknown_server.png";
/// `textures/gui/server_selection.png` — the join / move-up / move-down overlay buttons.
pub const SERVER_SELECTION_TEXTURE: &str = "textures/gui/server_selection.png";
/// `textures/gui/achievement/achievement_background.png` — the stats and achievements frame.
pub const ACHIEVEMENT_BACKGROUND_TEXTURE: &str =
    "textures/gui/achievement/achievement_background.png";
/// `textures/gui/container/stats_icons.png` — the item/block/mob icons on the stats screen.
pub const STATS_ICONS_TEXTURE: &str = "textures/gui/container/stats_icons.png";

/// The dirt background's tile size in GUI pixels. java: `GuiScreen.drawBackground` — `f = 32.0F`
pub const BACKGROUND_TILE: f32 = 32.0;

/// The tint vanilla multiplies the dirt background by: `color(64, 64, 64, 255)`.
pub const BACKGROUND_TINT: u32 = 0xFF40_4040;

/// The two stops of the translucent gradient drawn over a live world.
// java: GuiScreen.java:640 (drawWorldBackground) — -1072689136 and -804253680
pub const WORLD_BACKGROUND_TOP: u32 = 0xC010_1010;
/// Bottom stop of [`WORLD_BACKGROUND_TOP`]'s gradient.
pub const WORLD_BACKGROUND_BOTTOM: u32 = 0xD010_1010;

/// Vanilla's "title" white, `16777215`.
pub const TITLE_COLOR: i32 = 0x00FF_FFFF;
/// Vanilla's dimmed label grey, `10526880`.
pub const LABEL_COLOR: i32 = 10_526_880;
/// Vanilla's F3 / server-list text colour, `14737632`.
pub const INFO_COLOR: i32 = 14_737_632;

/// Tile `options_background.png` over the whole screen at `tint` rows of vertical offset.
///
/// This is the dirt every out-of-game screen sits on. `tint` shifts the texture down by whole
/// tiles — the scrolling list screens pass 0, and the *only* caller that passes non-zero is
/// `GuiWinGame`, which is single-player and therefore out of scope.
// java: GuiScreen.java:648 (drawBackground)
pub fn draw_dirt_background(d: &mut DrawList, width: i32, height: i32, tint: f32) {
    d.tiled(
        &UiTexture::gui(OPTIONS_BACKGROUND),
        Rect::new(0, 0, width, height),
        BACKGROUND_TILE,
        tint,
        BACKGROUND_TINT,
    );
}

/// Vanilla's `drawWorldBackground`: a translucent gradient when a world is loaded, the dirt
/// otherwise.
// java: GuiScreen.java:636
pub fn draw_world_background(d: &mut DrawList, width: i32, height: i32, in_world: bool) {
    if in_world {
        d.gradient_rect(
            0,
            0,
            width,
            height,
            WORLD_BACKGROUND_TOP,
            WORLD_BACKGROUND_BOTTOM,
        );
    } else {
        draw_dirt_background(d, width, height, 0.0);
    }
}

/// Whether the unicode page fonts should be used, given the active language and settings.
///
/// This is the second of the two gaps S15 handed S16: vanilla recomputes it in exactly two
/// places — at language switch and when `FORCE_UNICODE_FONT` is toggled — and both reduce to
/// this disjunction. Keeping it in one function means a screen can never set one and forget
/// the other.
// java: GameSettings.java:436 (`setOptionValue`, FORCE_UNICODE_FONT) and
// java: Minecraft.java:2321 (`refreshResources`)
#[must_use]
pub fn unicode_font_enabled(language: &Language, settings: &GameSettings) -> bool {
    language.is_unicode() || settings.force_unicode_font
}

/// Apply [`unicode_font_enabled`] to a font renderer, returning whether the flag changed.
///
/// Returns `true` when the caller must re-lay-out every open screen: glyph widths change with
/// the flag, so cached wraps and button labels are stale.
pub fn sync_unicode_flag(
    font: &mut FontRenderer,
    language: &Language,
    settings: &GameSettings,
) -> bool {
    let wanted = unicode_font_enabled(language, settings);
    let changed = font.unicode_flag() != wanted;
    font.set_unicode_flag(wanted);
    changed
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::screens::container::tooltip_box::draw as draw_hovering_text;
    use crate::widget::test_support::uniform_font;

    #[test]
    fn the_dirt_background_tiles_at_thirty_two_pixels() {
        let mut d = DrawList::new();
        draw_dirt_background(&mut d, 640, 480, 0.0);
        let v = d.vertices();
        assert_eq!(v.len(), 6);
        assert_eq!(v[0].pos, [0.0, 0.0]);
        assert_eq!(v[0].uv, [0.0, 0.0]);
        let br = v.iter().find(|v| v.pos == [640.0, 480.0]).expect("corner");
        assert_eq!(br.uv, [20.0, 15.0], "640/32 by 480/32 tiles");
        // The 0x404040 tint is what makes vanilla's menu dirt dark rather than block-bright.
        let c = 64.0 / 255.0;
        assert_eq!(v[0].color, [c, c, c, 1.0]);
    }

    #[test]
    fn a_loaded_world_swaps_the_dirt_for_the_translucent_gradient() {
        let mut world = DrawList::new();
        draw_world_background(&mut world, 100, 50, true);
        assert_eq!(world.batches().len(), 1);
        assert_eq!(world.batches()[0].texture, UiTexture::Solid);
        // Alpha 0xC0 at the top, 0xD0 at the bottom: the gradient really darkens downwards.
        let top = world
            .vertices()
            .iter()
            .find(|v| v.pos[1] == 0.0)
            .expect("top");
        let bottom = world
            .vertices()
            .iter()
            .find(|v| v.pos[1] == 50.0)
            .expect("bottom");
        assert!((top.color[3] - 192.0 / 255.0).abs() < 1e-6);
        assert!((bottom.color[3] - 208.0 / 255.0).abs() < 1e-6);

        let mut menu = DrawList::new();
        draw_world_background(&mut menu, 100, 50, false);
        assert!(matches!(menu.batches()[0].texture, UiTexture::Resource(_)));
    }

    #[test]
    fn the_first_tooltip_line_gets_vanillas_extra_two_pixel_step() {
        let mut font = uniform_font();
        let mut d = DrawList::new();
        let lines = vec!["a".to_owned(), "b".to_owned(), "c".to_owned()];
        draw_hovering_text(&mut d, &mut font, &lines, 100, 100, 640, 480);
        // Glyph quads are the only font-textured vertices; collect their distinct top rows.
        let mut rows: Vec<i32> = d
            .batches()
            .iter()
            .filter(|b| matches!(b.texture, UiTexture::Font(_)))
            .flat_map(|b| d.vertices()[b.range.clone()].iter())
            .map(|v| v.pos[1] as i32)
            .collect();
        rows.sort_unstable();
        rows.dedup();
        // Each line draws a shadow copy one pixel down and right, so every top row `t` also
        // produces `t + 1`. The line tops are 88, 88 + 2 + 10 = 100, and 110 — *not* 88/98/108,
        // which is what dropping vanilla's one-off `+2` would give.
        for expected in [88, 100, 110] {
            assert!(
                rows.contains(&expected),
                "no glyph row at {expected}: {rows:?}"
            );
        }
        assert!(
            !rows.contains(&98),
            "the extra two pixels after line 1 are missing"
        );
    }

    #[test]
    fn the_unicode_flag_is_the_language_or_the_forced_setting() {
        let mut settings = GameSettings::default();
        let mut ascii = Language::new("en_US");
        ascii.insert("k", "v");
        assert!(!ascii.is_unicode());

        assert!(!unicode_font_enabled(&ascii, &settings));
        settings.force_unicode_font = true;
        assert!(unicode_font_enabled(&ascii, &settings));

        settings.force_unicode_font = false;
        let mut cjk = Language::new("zh_CN");
        cjk.insert("k", "\u{4e00}\u{4e01}");
        cjk.recompute_unicode();
        assert!(cjk.is_unicode(), "a CJK-heavy language is unicode");
        assert!(unicode_font_enabled(&cjk, &settings));
    }

    #[test]
    fn syncing_the_flag_reports_whether_a_relayout_is_needed() {
        let mut font = uniform_font();
        let mut settings = GameSettings::default();
        let ascii = Language::new("en_US");
        assert!(
            !sync_unicode_flag(&mut font, &ascii, &settings),
            "no change"
        );
        settings.force_unicode_font = true;
        assert!(sync_unicode_flag(&mut font, &ascii, &settings));
        assert!(font.unicode_flag());
        assert!(
            !sync_unicode_flag(&mut font, &ascii, &settings),
            "idempotent"
        );
    }
}
