//! A resource pack's font sheet (`textures/font/ascii.png`) is decoded and measured cell by
//! cell. A very wide sheet still loads (a sheet over the device's texture limit is left blank at
//! upload: `sl_pack_texture_limits.rs`), so measuring one must not overflow.

use std::io::Cursor;

use mc_ui::font::{measure_ascii_widths, FontData};

fn png(width: u32, height: u32) -> Vec<u8> {
    let mut out = Cursor::new(Vec::new());
    image::RgbaImage::new(width, height)
        .write_to(&mut out, image::ImageFormat::Png)
        .expect("encode");
    out.into_inner()
}

#[test]
fn measuring_a_very_wide_sheet_does_not_overflow() {
    // Control: the last cell row's offset times the width, as the old `i32` index computed it.
    let (cell_width, width) = (49_152i32 / 16, 49_152i32);
    assert!((15 * cell_width).checked_mul(width).is_none());
    // No pixels: the measurement only reads what exists, so every cell is blank (one wide).
    let widths = measure_ascii_widths(&[], 49_152, 16);
    assert!(widths.iter().all(|&width| width == 1), "{:?}", &widths[..8]);
}

#[test]
fn a_very_wide_font_sheet_loads_and_is_measured() {
    // 49152 x 16: a few hundred bytes of PNG, 3 MiB of pixels, a 16x16 grid of cells.
    let data = FontData::from_parts(&png(49_152, 16), &[0u8; 0x1_0000]).expect("loads");
    assert_eq!(data.ascii_size(), (49_152, 16));
    assert!(data.char_widths().iter().all(|&width| width == 1));
}
