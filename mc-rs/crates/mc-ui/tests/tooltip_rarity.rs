use mc_ui::screens::container::tooltip::Rarity;
use mc_ui::screens::container::tooltip_box::recolour_tooltip_lines;

/// java: `GuiScreen.renderToolTip` prefixes the title with
/// `stack.getRarity().rarityColor` and every continuation with gray.
#[test]
fn render_tooltip_applies_every_rarity_colour_to_the_title() {
    for (rarity, code) in [
        (Rarity::Common, "\u{a7}f"),
        (Rarity::Uncommon, "\u{a7}e"),
        (Rarity::Rare, "\u{a7}b"),
        (Rarity::Epic, "\u{a7}d"),
    ] {
        let lines = recolour_tooltip_lines(vec!["Title".into(), "Detail".into()], rarity);
        assert_eq!(lines, [format!("{code}Title"), "\u{a7}7Detail".into()]);
    }
}
