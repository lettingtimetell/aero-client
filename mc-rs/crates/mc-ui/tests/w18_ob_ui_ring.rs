//! `MC_W18_OB_UI_RING`: the UI's vertex and
//! depth streams written straight into staging memory are the bytes the scratch-buffer path
//! uploads.
//!
//! ```text
//! cargo test -p mc-ui --test w18_ob_ui_ring
//! ```

use mc_ui::widget::draw::UiVertex;
use mc_ui::widget::{DrawList, Transform};

/// A list with varied geometry: rects, lines, a gradient, a transform, opacity and an item depth,
/// so positions, colours and depths all differ between vertices. `first` is the first rect's
/// colour.
fn sample_list(first: u32) -> DrawList {
    let mut list = DrawList::new();
    list.rect(0, 0, 20, 10, first);
    list.gradient_rect(3, 4, 40, 30, 0x8010_2030, 0xC0F0_E0D0);
    list.horizontal_line(1, 50, 7, 0xFFFF_FFFF);
    list.with_opacity(0.37, |list| {
        list.vertical_line(9, 2, 60, 0x7712_3456);
    });
    list.with_transform(Transform::translate(2.5, -1.25), |list| {
        list.rect(-5, -5, 5, 5, 0xFF00_FF00);
    });
    list.with_item_depth(0.625, |list| list.rect(10, 10, 12, 12, 0xFFAB_CDEF));
    list
}

fn staged_vertices(list: &DrawList) -> Vec<u8> {
    let mut staged = vec![0xAA_u8; list.vertices().len() * UiVertex::SIZE];
    list.fill_vertex_bytes(wgpu::WriteOnly::from_mut(staged.as_mut_slice()));
    staged
}

#[test]
fn the_staged_ui_streams_are_the_bytes_of_the_scratch_path() {
    let list = sample_list(0xFF20_4080);
    assert!(list.vertices().len() >= 24, "a list with several quads");

    let mut scratch = Vec::new();
    list.write_vertex_bytes(&mut scratch);
    assert_eq!(staged_vertices(&list), scratch, "vertex stream");

    // The staging memory starts as a sentinel no depth or vertex byte pattern here has, so an
    // unwritten byte would fail the comparison.
    let mut scratch = Vec::new();
    list.write_depth_bytes(&mut scratch);
    assert!(!scratch.is_empty() && scratch.iter().any(|&b| b != 0x55));
    let mut staged = vec![0x55_u8; list.depths().len() * 4];
    list.fill_depth_bytes(wgpu::WriteOnly::from_mut(staged.as_mut_slice()));
    assert_eq!(staged, scratch, "depth stream");

    // Negative control: one colour bit of the first rect is visible to the comparison.
    let other = sample_list(0xFF20_4081);
    assert_eq!(other.vertices().len(), list.vertices().len());
    let mut reference = Vec::new();
    list.write_vertex_bytes(&mut reference);
    assert_ne!(staged_vertices(&other), reference, "control");
}
