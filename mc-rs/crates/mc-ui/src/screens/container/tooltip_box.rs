//! The tooltip popup box: the dark panel plus its two-tone gradient border that vanilla draws
//! around [`super::tooltip::build`]'s lines. java: `GuiScreen.drawHoveringText`
//! (`GuiScreen.java:189`) and `GuiScreen.renderToolTip` (`GuiScreen.java:158`).
//!
//! [`super::tooltip`] produces the *text*; this module produces the *box*. They are separate
//! because vanilla separates them: `ItemStack.getTooltip` is pure data, and every caller
//! (an item slot, a creative tab label, an enchanting-table clue) hands its own list of lines to
//! the same `drawHoveringText`.
//!
//! # Layout, exactly as vanilla computes it
//!
//! The anchor is offset `(+12, -12)` from the mouse, not from the box's own corner, and the box
//! is then pushed back on-screen by two **asymmetric** rules that are reproduced verbatim in
//! [`layout`]:
//!
//! * horizontally: `if (x + 12 + width > screen_width) { x -= 28 + width }` — a 28-pixel jump,
//!   not a clamp, so a box that only just overflows lands well left of the cursor;
//! * vertically: `if (y - 12 + height + 6 > screen_height) { y = screen_height - height - 6 }` —
//!   a clamp, and one that can push the box *off the top* on a very short screen. Vanilla does
//!   not guard that and neither does this port.
//!
//! `height` above is vanilla's `k`: `8` for a single line, and `8 + 2 + (n - 1) * 10` for `n`
//! lines — i.e. the first line's baseline gap is 2 pixels bigger than every later one, which is
//! also why [`draw`]'s text loop advances by 12 after the first line and 10 after the rest.

use crate::font::FontRenderer;
use crate::widget::DrawList;

/// The tooltip panel fill, `-267386864`. java: `GuiScreen.java:227` (`int l = -267386864`).
pub const BACKGROUND: u32 = 0xF010_0010;

/// The border gradient's top colour, `1347420415`. java: `GuiScreen.java:236`
/// (`int i1 = 1347420415`).
pub const BORDER_TOP: u32 = 0x5050_00FF;

/// The border gradient's bottom colour: vanilla derives it from [`BORDER_TOP`] as
/// `(i1 & 16711422) >> 1 | i1 & -16777216` — halve every colour channel, keep the alpha.
/// java: `GuiScreen.java:237`.
pub const BORDER_BOTTOM: u32 = 0x5028_007F;

/// Vanilla's dimming gradient over the world behind an open container. java:
/// `GuiScreen.drawWorldBackground` (`GuiScreen.java:673`) — `-1072689136` to `-804253680`.
pub const WORLD_DIM_TOP: u32 = 0xC010_1010;
/// The bottom end of [`WORLD_DIM_TOP`]'s gradient.
pub const WORLD_DIM_BOTTOM: u32 = 0xD010_1010;

/// Where a tooltip box lands, before any drawing happens.
///
/// `x`/`y` are vanilla's `l1`/`i2`: the **text** origin, not the panel's top-left — the panel
/// extends 3–4 pixels further out on every side. `width` is the widest line and `height` is
/// vanilla's `k`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TooltipLayout {
    /// Left edge of the text, vanilla's `l1`.
    pub x: i32,
    /// Baseline top of the first line, vanilla's `i2`.
    pub y: i32,
    /// The widest line's pixel width, vanilla's `i`.
    pub width: i32,
    /// The text block's height, vanilla's `k`.
    pub height: i32,
}

impl TooltipLayout {
    /// The filled panel's outer bounds `(left, top, right, bottom)`, i.e. the union of the five
    /// [`BACKGROUND`] rectangles [`draw`] emits. Exclusive on right/bottom, like every other
    /// rectangle in this crate.
    #[must_use]
    pub const fn panel_bounds(&self) -> (i32, i32, i32, i32) {
        (
            self.x - 4,
            self.y - 4,
            self.x + self.width + 4,
            self.y + self.height + 4,
        )
    }
}

/// Vanilla's tooltip placement arithmetic, with no drawing. Returns `None` for an empty list,
/// matching `drawHoveringText`'s own `if (!textLines.isEmpty())` guard.
// java: GuiScreen.java:189 (drawHoveringText)
#[must_use]
pub fn layout(
    font: &FontRenderer,
    lines: &[String],
    mouse_x: i32,
    mouse_y: i32,
    screen_width: i32,
    screen_height: i32,
) -> Option<TooltipLayout> {
    if lines.is_empty() {
        return None;
    }
    let width = lines
        .iter()
        .map(|line| font.string_width(line))
        .max()
        .unwrap_or(0);

    let mut x = mouse_x + 12;
    let mut y = mouse_y - 12;
    let mut height = 8;
    if lines.len() > 1 {
        height += 2 + (lines.len() as i32 - 1) * 10;
    }
    if x + width > screen_width {
        x -= 28 + width;
    }
    if y + height + 6 > screen_height {
        y = screen_height - height - 6;
    }
    Some(TooltipLayout {
        x,
        y,
        width,
        height,
    })
}

/// Draw the popup box and its lines. No-op for an empty list.
///
/// The nine border/panel rectangles are emitted in vanilla's own order — five [`BACKGROUND`]
/// pieces (top strip, bottom strip, middle block, left strip, right strip), then four
/// [`BORDER_TOP`]/[`BORDER_BOTTOM`] gradient strips — because they overlap and the later ones
/// have to blend *over* the earlier ones to produce vanilla's exact corner colours.
// java: GuiScreen.java:189 (drawHoveringText)
pub fn draw(
    d: &mut DrawList,
    font: &mut FontRenderer,
    lines: &[String],
    mouse_x: i32,
    mouse_y: i32,
    screen_width: i32,
    screen_height: i32,
) {
    let Some(l) = layout(font, lines, mouse_x, mouse_y, screen_width, screen_height) else {
        return;
    };
    draw_layout(d, font, lines, &l);
}

/// [`draw`] at an already computed (possibly moved) [`TooltipLayout`]: the panel, the border and
/// the lines, in vanilla's order. [`draw`] is exactly [`layout`] followed by this; the
/// `scrollable_tooltips` mod (`super::tooltip_scroll`) shifts the layout first.
pub fn draw_layout(d: &mut DrawList, font: &mut FontRenderer, lines: &[String], l: &TooltipLayout) {
    let (x, mut y, w, h) = (l.x, l.y, l.width, l.height);

    d.gradient_rect(x - 3, y - 4, x + w + 3, y - 3, BACKGROUND, BACKGROUND);
    d.gradient_rect(
        x - 3,
        y + h + 3,
        x + w + 3,
        y + h + 4,
        BACKGROUND,
        BACKGROUND,
    );
    d.gradient_rect(x - 3, y - 3, x + w + 3, y + h + 3, BACKGROUND, BACKGROUND);
    d.gradient_rect(x - 4, y - 3, x - 3, y + h + 3, BACKGROUND, BACKGROUND);
    d.gradient_rect(
        x + w + 3,
        y - 3,
        x + w + 4,
        y + h + 3,
        BACKGROUND,
        BACKGROUND,
    );

    d.gradient_rect(x - 3, y - 2, x - 2, y + h + 2, BORDER_TOP, BORDER_BOTTOM);
    d.gradient_rect(
        x + w + 2,
        y - 2,
        x + w + 3,
        y + h + 2,
        BORDER_TOP,
        BORDER_BOTTOM,
    );
    d.gradient_rect(x - 3, y - 3, x + w + 3, y - 2, BORDER_TOP, BORDER_TOP);
    d.gradient_rect(
        x - 3,
        y + h + 2,
        x + w + 3,
        y + h + 3,
        BORDER_BOTTOM,
        BORDER_BOTTOM,
    );

    for (index, line) in lines.iter().enumerate() {
        font.draw_string_with_shadow(d, line, x as f32, y as f32, -1);
        if index == 0 {
            y += 2;
        }
        y += 10;
    }
}

/// Vanilla's `renderToolTip` line recolouring: line `0` is prefixed with the item's
/// `EnumRarity.rarityColor` and every later line with gray. Applied by
/// [`super::screen::ContainerScreen`] and [`super::creative_screen`] before [`draw`].
///
/// The rarity half used to be missing, with a comment saying "this port has no rarity table".
/// It looked harmless because an unprefixed first line renders in the caller's `-1`, which is
/// the same white as `COMMON` - so the one case that was right was right by accident, and
/// every enchanted item, every music disc, both golden apples and every enchanted book drew a
/// white title where vanilla draws aqua, light purple or yellow.
// java: GuiScreen.java:158-175 (renderToolTip)
#[must_use]
pub fn recolour_tooltip_lines(lines: Vec<String>, rarity: super::tooltip::Rarity) -> Vec<String> {
    let head = rarity.color_code();
    lines
        .into_iter()
        .enumerate()
        .map(|(i, line)| {
            if i == 0 {
                format!("{head}{line}")
            } else {
                format!("\u{a7}7{line}")
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::widget::test_support::uniform_font;
    use mc_nbt::{Nbt, NbtCompound};
    use mc_protocol::ItemStack;

    fn lines(n: usize) -> Vec<String> {
        (0..n).map(|i| format!("line{i}")).collect()
    }

    fn stack(name: &str) -> ItemStack {
        let id = mc_data::item_by_name(name).expect("test item is in the 1.8.9 registry");
        ItemStack {
            item_id: id.0 as i16,
            count: 1,
            damage: 0,
            nbt: None,
        }
    }

    fn title_after_render_tooltip(stack: &ItemStack) -> String {
        super::super::tooltip::build(stack, false)
            .into_iter()
            .next()
            .map(|title| {
                recolour_tooltip_lines(vec![title], super::super::tooltip::rarity(stack)).remove(0)
            })
            .expect("every item stack has a display-name line")
    }

    #[test]
    fn the_border_bottom_colour_is_vanillas_derived_half_brightness() {
        // java: `(i1 & 16711422) >> 1 | i1 & -16777216`, evaluated on BORDER_TOP.
        let derived = ((BORDER_TOP & 16_711_422) >> 1) | (BORDER_TOP & 0xFF00_0000);
        assert_eq!(derived, BORDER_BOTTOM);
    }

    #[test]
    fn the_packed_constants_match_the_javas_signed_ints() {
        assert_eq!(BACKGROUND as i32, -267_386_864);
        assert_eq!(BORDER_TOP as i32, 1_347_420_415);
        assert_eq!(WORLD_DIM_TOP as i32, -1_072_689_136);
        assert_eq!(WORLD_DIM_BOTTOM as i32, -804_253_680);
    }

    #[test]
    fn a_single_line_box_is_eight_tall_and_anchored_twelve_right_twelve_up() {
        let font = uniform_font();
        let l = layout(&font, &lines(1), 100, 100, 320, 240).unwrap();
        assert_eq!((l.x, l.y), (112, 88));
        assert_eq!(l.height, 8);
    }

    #[test]
    fn extra_lines_add_ten_each_plus_a_two_pixel_first_line_gap() {
        let font = uniform_font();
        assert_eq!(
            layout(&font, &lines(1), 0, 100, 320, 240).unwrap().height,
            8
        );
        assert_eq!(
            layout(&font, &lines(2), 0, 100, 320, 240).unwrap().height,
            20
        );
        assert_eq!(
            layout(&font, &lines(3), 0, 100, 320, 240).unwrap().height,
            30
        );
    }

    #[test]
    fn an_empty_list_lays_out_and_draws_nothing() {
        let mut font = uniform_font();
        assert_eq!(layout(&font, &[], 10, 10, 320, 240), None);
        let mut d = DrawList::new();
        draw(&mut d, &mut font, &[], 10, 10, 320, 240);
        assert!(d.is_empty());
    }

    /// Vanilla does not clamp horizontally; it *jumps* by `28 + width`, which is a different
    /// x than a clamp would produce and is exactly what makes a tooltip near the right edge
    /// appear to the left of the cursor with a visible gap.
    #[test]
    fn overflowing_the_right_edge_jumps_the_box_left_rather_than_clamping_it() {
        let font = uniform_font();
        let text = vec!["wide".to_string()];
        let w = font.string_width("wide");
        let l = layout(&font, &text, 310, 100, 320, 240).unwrap();
        assert_eq!(l.x, 310 + 12 - 28 - w);
        assert!(
            l.x < 320 - w,
            "the jumped box still has to fit: {} vs {}",
            l.x,
            320 - w
        );
    }

    #[test]
    fn a_box_that_fits_is_not_moved_horizontally() {
        let font = uniform_font();
        let l = layout(&font, &lines(1), 10, 100, 320, 240).unwrap();
        assert_eq!(l.x, 22);
    }

    #[test]
    fn overflowing_the_bottom_edge_clamps_the_box_up() {
        let font = uniform_font();
        let l = layout(&font, &lines(3), 100, 238, 320, 240).unwrap();
        assert_eq!(l.y, 240 - 30 - 6);
    }

    /// A pinned vanilla bug: the vertical rule is a bare assignment, so on a screen shorter than
    /// the tooltip the box is pushed above `y = 0` and its top is simply clipped away. Vanilla
    /// has no guard here; reproducing it keeps a parity trace comparable.
    #[test]
    fn a_tooltip_taller_than_the_screen_is_pushed_off_the_top_exactly_like_vanilla() {
        let font = uniform_font();
        let l = layout(&font, &lines(6), 10, 10, 320, 20).unwrap();
        assert_eq!(l.height, 8 + 2 + 5 * 10);
        assert_eq!(l.y, 20 - l.height - 6);
        assert!(l.y < 0, "vanilla lets this go negative: {}", l.y);
    }

    #[test]
    fn the_panel_covers_four_pixels_beyond_the_text_on_every_side() {
        let font = uniform_font();
        let l = layout(&font, &lines(1), 100, 100, 320, 240).unwrap();
        let (left, top, right, bottom) = l.panel_bounds();
        assert_eq!(left, l.x - 4);
        assert_eq!(top, l.y - 4);
        assert_eq!(right, l.x + l.width + 4);
        assert_eq!(bottom, l.y + l.height + 4);
    }

    #[test]
    fn drawing_emits_nine_panel_pieces_before_any_text() {
        let mut font = uniform_font();
        let mut d = DrawList::new();
        draw(&mut d, &mut font, &lines(1), 100, 100, 320, 240);
        // Nine solid quads = 54 vertices, then the glyphs.
        assert!(d.vertices().len() > 54);
        assert_eq!(
            d.batches()[0].range.len(),
            54,
            "the nine box pieces share the Solid texture and merge into one batch"
        );
    }

    #[test]
    fn every_box_piece_lands_inside_the_panel_bounds() {
        let mut font = uniform_font();
        let mut d = DrawList::new();
        draw(&mut d, &mut font, &lines(2), 100, 100, 320, 240);
        let l = layout(&uniform_font(), &lines(2), 100, 100, 320, 240).unwrap();
        let (left, top, right, bottom) = l.panel_bounds();
        for v in &d.vertices()[..54] {
            assert!(v.pos[0] >= left as f32 && v.pos[0] <= right as f32, "{v:?}");
            assert!(v.pos[1] >= top as f32 && v.pos[1] <= bottom as f32, "{v:?}");
        }
    }

    /// java: `renderToolTip` prefixes line `0` with `getRarity().rarityColor` and every later
    /// line with `GRAY` - the first line is **not** left alone, which is what this test used to
    /// assert.
    #[test]
    fn the_first_line_takes_the_rarity_colour_and_the_rest_are_forced_gray() {
        use super::super::tooltip::Rarity;
        let out = recolour_tooltip_lines(
            vec!["Name".into(), "Lore".into(), "More".into()],
            Rarity::Common,
        );
        assert_eq!(out[0], "\u{a7}fName");
        assert_eq!(out[1], "\u{a7}7Lore");
        assert_eq!(out[2], "\u{a7}7More");

        for (rarity, code) in [
            (Rarity::Uncommon, "\u{a7}e"),
            (Rarity::Rare, "\u{a7}b"),
            (Rarity::Epic, "\u{a7}d"),
        ] {
            let out = recolour_tooltip_lines(vec!["Name".into()], rarity);
            assert_eq!(out[0], format!("{code}Name"));
        }
    }

    /// `Item.getRarity` turns any `ench` list into `RARE`, even an empty one; `GuiScreen`
    /// applies that result after `ItemStack.getTooltip` has constructed the title. The empty-list
    /// control matters because testing only a populated list would fail to catch a port that
    /// incorrectly inspected the enchantments rather than `hasKey("ench", 9)`.
    #[test]
    fn ordinary_enchanted_item_titles_are_aqua_even_with_an_empty_ench_list() {
        let mut sword = stack("diamond_sword");
        let mut tag = NbtCompound::new();
        tag.insert("ench", Nbt::list(vec![]));
        sword.nbt = Some(tag);

        assert!(title_after_render_tooltip(&sword).starts_with("\u{a7}b"));
    }

    /// Java's three item overrides are observable at this seam: apples branch on metadata,
    /// enchanted books on a nonempty `StoredEnchantments` list, and `HideFlags` affects only
    /// `getTooltip`'s later sections -- never the `getRarity` title prefix in `renderToolTip`.
    #[test]
    fn item_overrides_metadata_and_hide_flags_reach_the_rendered_title_colour() {
        let apple = stack("golden_apple");
        assert!(title_after_render_tooltip(&apple).starts_with("\u{a7}b"));

        let mut enchanted_apple = stack("golden_apple");
        enchanted_apple.damage = 1;
        assert!(title_after_render_tooltip(&enchanted_apple).starts_with("\u{a7}d"));

        let mut book = stack("enchanted_book");
        let mut book_tag = NbtCompound::new();
        book_tag.insert(
            "StoredEnchantments",
            Nbt::list(vec![Nbt::Compound(NbtCompound::new())]),
        );
        book.nbt = Some(book_tag);
        assert!(title_after_render_tooltip(&book).starts_with("\u{a7}e"));

        let mut hidden = book.clone();
        hidden
            .nbt
            .as_mut()
            .expect("test book has a tag")
            .insert("HideFlags", Nbt::Int(127));
        assert_eq!(
            title_after_render_tooltip(&hidden),
            title_after_render_tooltip(&book),
            "HideFlags does not suppress GuiScreen's first-line rarity formatting"
        );
    }

    /// `ItemMap.addInformation` returns unformatted English literals; both vanilla tooltip
    /// callers then run the complete list through `GuiScreen.renderToolTip`, which makes the
    /// title the item's rarity colour and every map-information line gray. Keep this at the box
    /// seam so a future caller cannot accidentally leave the map lines white or colour them as a
    /// second title.
    #[test]
    fn filled_map_information_is_ordered_and_recoloured_at_the_render_seam() {
        use super::super::tooltip::{self, Rarity};
        use mc_world::MapData;

        let mut map = stack("filled_map");
        map.damage = 37;
        let mut data = MapData::new(37);
        data.scale = 3;

        let raw = tooltip::build_with_map_data(&map, true, Some(&data));
        assert_eq!(
            raw.len(),
            4,
            "title, ItemMap's two lines, and advanced id: {raw:?}"
        );
        assert_eq!(&raw[1..3], ["Scaling at 1:8", "(Level 3/4)"]);
        assert!(
            raw[1..3].iter().all(|line| !line.starts_with('\u{a7}')),
            "ItemStack.getTooltip itself returns the map literals unformatted"
        );

        let rendered = recolour_tooltip_lines(raw, Rarity::Common);
        assert!(rendered[0].starts_with("\u{a7}f"));
        assert_eq!(
            &rendered[1..3],
            ["\u{a7}7Scaling at 1:8", "\u{a7}7(Level 3/4)"]
        );
    }
}
