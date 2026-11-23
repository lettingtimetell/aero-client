//! Filled-map texture conversion and map-item geometry.
//!
//! java: `client/gui/MapItemRenderer`.  [`mc_world::MapData`] owns the received
//! `S34PacketMaps` image; this module is deliberately the next, GPU-free step.  A client app
//! uploads [`MapTexture::rgba`] under a dynamic [`UiTexture`] and calls
//! [`draw_map`] wherever an item model needs its live contents (GUI, held item, or item frame).
//! Keeping those two operations separate means the pixel and decoration rules remain fully
//! testable without a graphics adapter.

use std::collections::HashMap;

use mc_world::{MapData, MapDecoration, MAP_PIXELS, MAP_SIZE};

use crate::widget::{DrawList, Transform, UiTexture};

/// The side length of a vanilla filled-map image, in texels.
pub const MAP_TEXTURE_SIZE: usize = MAP_SIZE;

/// Bytes in a dynamic RGBA8 map texture.
pub const MAP_TEXTURE_BYTES: usize = MAP_PIXELS * 4;

/// The vanilla decoration sprite sheet.  It is a 4 by 4 sheet, not a 16-wide strip.
pub const MAP_ICONS_TEXTURE: &str = "textures/map/map_icons.png";

/// The `TextureManager.getDynamicTextureLocation("map", ...)`-style key for `map_id`.
///
/// Before submitting a list containing this key, upload [`MapTexture::rgba`] through
/// [`crate::widget::UiRenderer::upload`].  The normal resource-pack loader deliberately has no
/// file at this location: it is an in-memory `DynamicTexture`, like vanilla's `map_0`, `map_1`,
/// and so on.
#[must_use]
pub fn dynamic_texture_key(map_id: i32) -> UiTexture {
    UiTexture::gui(&format!("dynamic/map_{map_id}"))
}

/// `MapColor.mapColorArray`'s populated 1.8.9 entries, indexed by `colour_byte / 4`.
///
/// Entries 36 through 63 of Java's 64-slot table are null.  A normal server never sends a
/// colour byte selecting one of them, because it constructs the byte from this same palette.
const MAP_PALETTE: [u32; 36] = [
    0, 8_368_696, 16_247_203, 13_092_807, 16_711_680, 10_526_975, 10_987_431, 31_744, 16_777_215,
    10_791_096, 9_923_917, 7_368_816, 4_210_943, 9_402_184, 16_776_437, 14_188_339, 11_685_080,
    6_724_056, 15_066_419, 8_375_321, 15_892_389, 5_000_268, 10_066_329, 5_013_401, 8_339_378,
    3_361_970, 6_704_179, 6_717_235, 10_040_115, 1_644_825, 16_445_005, 6_085_589, 4_882_687,
    55_610, 8_476_209, 7_340_544,
];

/// A map colour byte selected a palette slot that Java would find null.
///
/// This intentionally does not silently substitute a colour: `MapItemRenderer` dereferences
/// the slot, so an invalid server-created colour is not equivalent to unexplored terrain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InvalidMapColor {
    /// Offset within the 128 by 128 colour image.
    pub pixel: usize,
    /// The received `MapData.colors[pixel]` byte.
    pub color: u8,
}

/// CPU pixels ready for one `Rgba8Unorm` dynamic texture upload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MapTexture {
    /// The `map_<id>` value which identifies this texture's source data.
    pub map_id: i32,
    /// Row-major RGBA8 texels, exactly 128 by 128.
    pub rgba: [u8; MAP_TEXTURE_BYTES],
}

impl MapTexture {
    /// Resolves one received map image through `MapColor.getMapColor`'s four shade ramp.
    pub fn from_data(data: &MapData) -> Result<Self, InvalidMapColor> {
        let mut rgba = [0; MAP_TEXTURE_BYTES];
        for (pixel, color) in data.colors.iter().copied().enumerate() {
            let texel = map_texel(pixel, color)?;
            rgba[pixel * 4..pixel * 4 + 4].copy_from_slice(&texel);
        }
        Ok(Self {
            map_id: data.id,
            rgba,
        })
    }
}

/// Retained dynamic map textures, keyed by the client-side `map_<id>` record.
///
/// This is the CPU half of `MapItemRenderer.field_148252_c`: vanilla creates one
/// `Instance` the first time a map name is seen, then `updateMapTexture` replaces that
/// instance's 128-by-128 dynamic texture on each subsequent S34 update.  The app owns the
/// GPU upload, but keeping the latest converted pixels here makes that upload a direct,
/// deterministic handoff rather than a second conversion implementation.
///
/// Each image lives in its own box: stored inline, every bucket of the
/// table (used or not) cost 64 KiB, so past 56 maps the table took an 8 MiB block (16 MiB past
/// 112) and kept it for the session. Now a bucket is a pointer, an update of a known map writes
/// into its box (no allocation), and [`Self::clear`] gives the table back.
#[derive(Debug, Default)]
pub struct MapTextureCache {
    textures: HashMap<i32, Box<MapTexture>>,
}

impl MapTextureCache {
    /// Convert and retain the current image for `data.id`.
    ///
    /// Conversion happens before the previous snapshot is replaced.  Therefore an invalid
    /// colour byte leaves an already-renderable map untouched.  This is the observable Java
    /// update boundary: `MapItemRenderer.Instance.updateMapTexture` mutates its CPU-side
    /// `mapTextureData` until `MapColor.mapColorArray[j / 4].getMapColor(...)` dereferences a
    /// null slot, but it does not reach `DynamicTexture.updateDynamicTexture()`.  The previously
    /// uploaded GPU image consequently remains visible; retaining the last successful complete
    /// snapshot is the safe Rust representation of that stale dynamic texture.  A map that has
    /// never uploaded successfully remains absent until a later valid update recovers it.
    pub fn update(&mut self, data: &MapData) -> Result<&MapTexture, InvalidMapColor> {
        let texture = MapTexture::from_data(data)?;
        // A map seen before is overwritten in its box (the same copy the inline value took); only
        // a new map allocates.
        let retained = match self.textures.entry(texture.map_id) {
            std::collections::hash_map::Entry::Occupied(entry) => {
                let retained = entry.into_mut();
                **retained = texture;
                retained
            }
            std::collections::hash_map::Entry::Vacant(entry) => entry.insert(Box::new(texture)),
        };
        Ok(&**retained)
    }

    /// The latest converted texture for `map_id`, if that map has been received.
    #[must_use]
    pub fn get(&self, map_id: i32) -> Option<&MapTexture> {
        self.textures.get(&map_id).map(|texture| &**texture)
    }

    /// Forget all retained dynamic textures, as vanilla does when its texture manager reloads.
    /// The table itself is given back too: this runs when a world is left, never per frame.
    pub fn clear(&mut self) {
        self.textures = HashMap::new();
    }

    /// Number of retained map records.
    #[must_use]
    pub fn len(&self) -> usize {
        self.textures.len()
    }

    /// Whether no map record has been retained.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.textures.is_empty()
    }
}

/// Resolve one `MapData.colors` byte to the texture's RGBA8 texel.
///
/// `colour / 4 == 0` is the low-alpha black checkerboard that Java uses for unexplored map
/// pixels.  All other values select `MapColor.mapColorArray[colour / 4]` and use shade 0/1/2/3
/// multipliers 180/220/255/135 respectively.
pub fn map_texel(pixel: usize, color: u8) -> Result<[u8; 4], InvalidMapColor> {
    let palette_index = usize::from(color / 4);
    if palette_index == 0 {
        let alpha = if ((pixel + pixel / MAP_SIZE) & 1) == 0 {
            16
        } else {
            24
        };
        return Ok([0, 0, 0, alpha]);
    }

    let Some(rgb) = MAP_PALETTE.get(palette_index).copied() else {
        return Err(InvalidMapColor { pixel, color });
    };
    let shade = match color & 3 {
        0 => 180,
        1 => 220,
        2 => 255,
        3 => 135,
        _ => unreachable!("the mask yields 0..=3"),
    };
    Ok([
        (((rgb >> 16) & 255) * shade / 255) as u8,
        (((rgb >> 8) & 255) * shade / 255) as u8,
        ((rgb & 255) * shade / 255) as u8,
        255,
    ])
}

/// Draw the live 128 by 128 texture and its ordered decorations.
///
/// `origin` is the top-left of the map. `scale` is GUI pixels per map texel: use `1.0` for the
/// native renderer quad or `1.0 / 8.0` for a 16 by 16 inventory slot. `map_texture` must name
/// the dynamic texture uploaded from [`MapTexture::rgba`].
///
/// With `no_overlay_rendering`, Java still draws icon type 1, but skips every other type.
/// This is how an item frame suppresses player markers while retaining the frame marker.
pub fn draw_map(
    draw: &mut DrawList,
    map_texture: &UiTexture,
    icons_texture: &UiTexture,
    origin: [f32; 2],
    scale: f32,
    data: &MapData,
    no_overlay_rendering: bool,
) {
    draw.quad_rect(
        map_texture,
        (
            origin[0],
            origin[1],
            origin[0] + MAP_SIZE as f32 * scale,
            origin[1] + MAP_SIZE as f32 * scale,
        ),
        (0.0, 0.0, 1.0, 1.0),
        0xFFFF_FFFF,
    );

    for decoration in &data.decorations {
        if no_overlay_rendering && decoration.icon_type != 1 {
            continue;
        }
        draw_decoration(draw, icons_texture, origin, scale, decoration);
    }
}

fn draw_decoration(
    draw: &mut DrawList,
    icons_texture: &UiTexture,
    origin: [f32; 2],
    scale: f32,
    decoration: &MapDecoration,
) {
    // MapItemRenderer: translate(x + b / 2 + 64, y + c / 2 + 64), rotate(direction * 360/16),
    // scale(4), translate(-.125, .125), then a -1..1 quad.  The dynamic map quad is 2-D here,
    // so its tiny negative Z offsets become submission order; DrawList preserves that order.
    let transform = Transform::translate(
        origin[0] + (f32::from(decoration.x) / 2.0 + 64.0) * scale,
        origin[1] + (f32::from(decoration.z) / 2.0 + 64.0) * scale,
    )
    .then_rotate_degrees(f32::from(decoration.direction) * 360.0 / 16.0)
    .then_scale(4.0 * scale)
    .then_translate(-0.125, 0.125);
    let col = f32::from(decoration.icon_type % 4) / 4.0;
    let row = f32::from(decoration.icon_type / 4) / 4.0;
    draw.with_transform(transform, |draw| {
        draw.quad_rect(
            icons_texture,
            (-1.0, -1.0, 1.0, 1.0),
            (col, row, col + 0.25, row + 0.25),
            0xFFFF_FFFF,
        );
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::widget::{UiTexture, UiVertex};
    use mc_protocol::play::s2c::{MapIcon, Maps};

    fn map() -> MapData {
        let mut data = MapData::new(41);
        data.apply_packet(&Maps {
            map_id: 41,
            scale: 0,
            icons: vec![
                MapIcon {
                    icon_type: 0,
                    x: 0,
                    z: 0,
                    direction: 0,
                },
                MapIcon {
                    icon_type: 5,
                    x: 8,
                    z: -4,
                    direction: 4,
                },
                MapIcon {
                    icon_type: 1,
                    x: -10,
                    z: 12,
                    direction: 12,
                },
            ],
            update: None,
        })
        .unwrap();
        data
    }

    fn bounds(vertices: &[UiVertex]) -> (f32, f32, f32, f32) {
        let min_x = vertices
            .iter()
            .map(|v| v.pos[0])
            .fold(f32::INFINITY, f32::min);
        let min_y = vertices
            .iter()
            .map(|v| v.pos[1])
            .fold(f32::INFINITY, f32::min);
        let max_x = vertices
            .iter()
            .map(|v| v.pos[0])
            .fold(f32::NEG_INFINITY, f32::max);
        let max_y = vertices
            .iter()
            .map(|v| v.pos[1])
            .fold(f32::NEG_INFINITY, f32::max);
        (min_x, min_y, max_x, max_y)
    }

    #[test]
    fn checkerboard_and_palette_shades_match_map_color() {
        assert_eq!(map_texel(0, 0).unwrap(), [0, 0, 0, 16]);
        assert_eq!(map_texel(1, 0).unwrap(), [0, 0, 0, 24]);
        assert_eq!(map_texel(128, 0).unwrap(), [0, 0, 0, 24]);
        // Palette index 8 is snow white. The four shades therefore expose the exact factors.
        assert_eq!(map_texel(9, 8 * 4).unwrap(), [180, 180, 180, 255]);
        assert_eq!(map_texel(9, 8 * 4 + 1).unwrap(), [220, 220, 220, 255]);
        assert_eq!(map_texel(9, 8 * 4 + 2).unwrap(), [255, 255, 255, 255]);
        assert_eq!(map_texel(9, 8 * 4 + 3).unwrap(), [135, 135, 135, 255]);
    }

    #[test]
    fn every_populated_palette_slot_matches_java_map_color() {
        let java = [
            0, 8_368_696, 16_247_203, 13_092_807, 16_711_680, 10_526_975, 10_987_431, 31_744,
            16_777_215, 10_791_096, 9_923_917, 7_368_816, 4_210_943, 9_402_184, 16_776_437,
            14_188_339, 11_685_080, 6_724_056, 15_066_419, 8_375_321, 15_892_389, 5_000_268,
            10_066_329, 5_013_401, 8_339_378, 3_361_970, 6_704_179, 6_717_235, 10_040_115,
            1_644_825, 16_445_005, 6_085_589, 4_882_687, 55_610, 8_476_209, 7_340_544,
        ];
        assert_eq!(MAP_PALETTE, java);

        for color in 0..=3 {
            assert!(map_texel(0, color).is_ok(), "unexplored shade {color}");
        }
        for slot in 36_u8..64 {
            for shade in 0..4 {
                assert!(
                    map_texel(0, slot * 4 + shade).is_err(),
                    "null Java palette slot {slot}, shade {shade}"
                );
            }
        }
    }

    #[test]
    fn null_palette_boundary_bytes_fail_without_wrapping_or_panicking() {
        // Java's 64-slot table has null entries at 36..63.  The first and last encoded bytes
        // are the important u8 boundaries: 36 * 4 == 144 and 63 * 4 + 3 == 255.
        assert_eq!(
            map_texel(11, 144),
            Err(InvalidMapColor {
                pixel: 11,
                color: 144,
            })
        );
        assert_eq!(
            map_texel(12, 255),
            Err(InvalidMapColor {
                pixel: 12,
                color: 255,
            })
        );
        assert!(map_texel(13, 35 * 4 + 3).is_ok());
    }

    #[test]
    fn unknown_palette_slot_is_not_mistaken_for_unexplored() {
        assert_eq!(
            map_texel(77, 36 * 4),
            Err(InvalidMapColor {
                pixel: 77,
                color: 36 * 4,
            })
        );
    }

    #[test]
    fn texture_uses_row_major_map_data_bytes() {
        let mut data = MapData::new(-7);
        data.colors[2 + 3 * MAP_SIZE] = 8 * 4 + 2;
        let texture = MapTexture::from_data(&data).unwrap();
        assert_eq!(texture.map_id, -7);
        assert_eq!(
            dynamic_texture_key(texture.map_id),
            UiTexture::gui("dynamic/map_-7")
        );
        assert_eq!(
            texture.rgba[(2 + 3 * MAP_SIZE) * 4..][..4],
            [255, 255, 255, 255]
        );
        assert_eq!(texture.rgba[(3 + 2 * MAP_SIZE) * 4..][..4], [0, 0, 0, 24]);
    }

    #[test]
    fn cache_replaces_one_map_snapshot_without_disturbing_another() {
        let mut first = MapData::new(4);
        first.colors[0] = 8 * 4 + 2;
        let mut second = MapData::new(-9);
        second.colors[7] = 8 * 4 + 3;

        let mut cache = MapTextureCache::default();
        cache.update(&first).unwrap();
        cache.update(&second).unwrap();
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.get(4).unwrap().rgba[..4], [255, 255, 255, 255]);
        assert_eq!(
            cache.get(-9).unwrap().rgba[7 * 4..][..4],
            [135, 135, 135, 255]
        );

        // `MapItemRenderer` retains one Instance per map name.  A later S34 for map_4 must
        // update that texture only, not recreate or contaminate the map_-9 record.
        first.colors[0] = 8 * 4;
        cache.update(&first).unwrap();
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.get(4).unwrap().rgba[..4], [180, 180, 180, 255]);
        assert_eq!(
            cache.get(-9).unwrap().rgba[7 * 4..][..4],
            [135, 135, 135, 255]
        );

        cache.clear();
        assert!(cache.is_empty());
    }

    #[test]
    fn failed_cache_update_preserves_the_last_renderable_snapshot() {
        let mut data = MapData::new(12);
        data.colors[1] = 8 * 4 + 1;
        let mut cache = MapTextureCache::default();
        cache.update(&data).unwrap();
        let before = cache.get(12).unwrap().clone();

        data.colors[1] = 36 * 4;
        assert_eq!(
            cache.update(&data),
            Err(InvalidMapColor {
                pixel: 1,
                color: 36 * 4,
            })
        );
        assert_eq!(cache.get(12), Some(&before));
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn malformed_first_update_has_no_texture_then_recovers_atomically() {
        let mut data = MapData::new(13);
        data.colors[0] = 36 * 4;
        let mut cache = MapTextureCache::default();

        // A Java null-slot dereference happens before the first dynamic-texture upload.  Rust
        // therefore keeps no renderable snapshot for a map that has never succeeded.
        assert_eq!(
            cache.update(&data),
            Err(InvalidMapColor {
                pixel: 0,
                color: 36 * 4,
            })
        );
        assert!(cache.get(13).is_none());
        assert!(cache.is_empty());

        // Once the bad byte is corrected, the complete 128x128 image uploads as one replacement.
        data.colors[0] = 35 * 4 + 3;
        let recovered = cache.update(&data).unwrap();
        assert_eq!(recovered.rgba[..4], map_texel(0, 35 * 4 + 3).unwrap());
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn decorations_follow_packet_order_and_four_by_four_atlas_layout() {
        let data = map();
        let map_tex = UiTexture::gui("dynamic/map_41");
        let icons = UiTexture::gui(MAP_ICONS_TEXTURE);
        let mut draw = DrawList::new();
        draw_map(&mut draw, &map_tex, &icons, [10.0, 20.0], 1.0, &data, false);

        assert_eq!(draw.vertices().len(), 24, "map then three icons");
        assert_eq!(draw.batches().len(), 2);
        assert_eq!(draw.batches()[0].texture, map_tex);
        assert_eq!(draw.batches()[1].texture, icons);
        // Icon 0 is centred on the map's centre and its -1..1 quad becomes 8x8 at scale 1.
        assert_eq!(bounds(&draw.vertices()[6..12]), (69.5, 80.5, 77.5, 88.5));
        // Icon 5 is row 1 / column 1 (not a 16-wide atlas), and direction 4 rotates 90 degrees.
        assert!(draw.vertices()[12..18]
            .iter()
            .all(|v| v.uv[0] >= 0.25 && v.uv[0] <= 0.5 && v.uv[1] >= 0.25 && v.uv[1] <= 0.5));
        assert_eq!(bounds(&draw.vertices()[12..18]), (73.5, 77.5, 81.5, 85.5));
    }

    #[test]
    fn no_overlay_rendering_retains_only_the_frame_marker() {
        let data = map();
        let mut draw = DrawList::new();
        draw_map(
            &mut draw,
            &UiTexture::gui("dynamic/map_41"),
            &UiTexture::gui(MAP_ICONS_TEXTURE),
            [0.0, 0.0],
            1.0 / 8.0,
            &data,
            true,
        );
        assert_eq!(draw.vertices().len(), 12, "map plus only type-1 marker");
        // 128 source texels at 1/8 produces the normal 16 by 16 item slot picture.
        assert_eq!(bounds(&draw.vertices()[0..6]), (0.0, 0.0, 16.0, 16.0));
    }
}
