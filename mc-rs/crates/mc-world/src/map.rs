//! Client-side map state fed by `S34PacketMaps`.
//!
//! java: `world/storage/MapData`'s client-reachable fields and
//! `S34PacketMaps.setMapdataTo`. The server's map sampling, dirty-rectangle production, and
//! persistence are deliberately not here: a multiplayer client only receives rectangles and
//! retains the resulting 128 by 128 colour-index image in memory.

use mc_protocol::play::s2c;

/// Width and height of a vanilla map image, in pixels.
pub const MAP_SIZE: usize = 128;

/// Number of colour-index bytes in one vanilla map image.
pub const MAP_PIXELS: usize = MAP_SIZE * MAP_SIZE;

/// A named decoration from [`MapData::decorations`].
///
/// Java stores these in `MapData.mapDecorations` under the keys `"icon-0"`, `"icon-1"`, ...
/// when handling a map packet. A vector preserves that packet order, which is also the draw
/// order the map renderer needs; `key` retains the Java-visible identity for callers that need
/// to address a particular decoration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MapDecoration {
    /// Java's `"icon-<packet index>"` key.
    pub key: String,
    /// The four-bit map-icon type.
    pub icon_type: u8,
    /// Horizontal map coordinate.
    pub x: i8,
    /// Vertical/world-Z-derived map coordinate.
    pub z: i8,
    /// The four-bit rotation.
    pub direction: u8,
}

impl From<(usize, &s2c::MapIcon)> for MapDecoration {
    fn from((index, icon): (usize, &s2c::MapIcon)) -> Self {
        Self {
            key: format!("icon-{index}"),
            icon_type: icon.icon_type,
            x: icon.x,
            z: icon.z,
            direction: icon.direction,
        }
    }
}

/// The client-reachable state of one filled map.
///
/// This is the in-memory `MapData` created by `ItemMap.loadMapData("map_" + id, world)` in
/// `NetHandlerPlayClient.handleMaps`. `x_center`, `z_center`, and `dimension` are not sent by
/// the 1.8.9 `S34PacketMaps` wire format, so they are intentionally absent; the renderer needs
/// only [`scale`](Self::scale), [`colors`](Self::colors), and [`decorations`](Self::decorations).
#[derive(Debug, Clone)]
pub struct MapData {
    /// `S34PacketMaps.mapId`; the backing Java storage name is `map_<id>`.
    pub id: i32,
    /// The packet's signed scale byte. Java copies it verbatim; it is not clamped here.
    pub scale: i8,
    /// The 128 x 128 colour-index image, addressed as `x + y * 128`.
    pub colors: [u8; MAP_PIXELS],
    /// The packet's current decorations in packet/draw order.
    pub decorations: Vec<MapDecoration>,
    /// A process-wide stamp of [`colors`](Self::colors): `0` for a blank
    /// [`MapData::new`] map, then a fresh, strictly larger value from every applied pixel
    /// rectangle. Renderers keep the revision they uploaded
    /// and re-upload only when it moves, which is vanilla's cadence:
    /// `MapItemRenderer.Instance.updateMapTexture` runs from the S34 handler, never per frame.
    /// Values are unique across worlds, so a reused map id after a server switch never matches
    /// an old upload. Not part of equality.
    pub revision: u64,
}

/// Equality is the client-visible map state only; [`MapData::revision`] records history.
impl PartialEq for MapData {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
            && self.scale == other.scale
            && self.colors == other.colors
            && self.decorations == other.decorations
    }
}

impl Eq for MapData {}

/// The source of [`MapData::revision`]. Starts at 1, so `0` always means "blank".
static NEXT_MAP_REVISION: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

impl MapData {
    /// Creates a blank in-memory map under `map_<id>`.
    #[must_use]
    pub fn new(id: i32) -> Self {
        Self {
            id,
            scale: 0,
            colors: [0; MAP_PIXELS],
            decorations: Vec::new(),
            revision: 0,
        }
    }

    /// Applies `S34PacketMaps.setMapdataTo`'s client-visible mutation.
    ///
    /// Scale and decorations are replaced on *every* packet, including one with no pixel
    /// rectangle. A present rectangle is copied in Java's order:
    /// `colors[min_x + x + (min_y + y) * 128] = data[x + y * width]`.
    ///
    /// `Maps::read` deliberately preserves the wire fields even when an untrusted peer gives an
    /// impossible rectangle. This state API rejects that malformed shape before it can index an
    /// array; normal protocol-47 packets take the exact Java branch described above.
    pub fn apply_packet(&mut self, packet: &s2c::Maps) -> Result<(), MapUpdateError> {
        let Some(update) = &packet.update else {
            self.scale = packet.scale;
            self.replace_decorations(&packet.icons);
            return Ok(());
        };

        let width = usize::from(update.width);
        let height = usize::from(update.height);
        let min_x = usize::from(update.min_x);
        let min_y = usize::from(update.min_y);
        let required = width.saturating_mul(height);
        if min_x.saturating_add(width) > MAP_SIZE || min_y.saturating_add(height) > MAP_SIZE {
            return Err(MapUpdateError::RectangleOutOfBounds {
                width: update.width,
                height: update.height,
                min_x: update.min_x,
                min_y: update.min_y,
            });
        }
        if update.data.len() < required {
            return Err(MapUpdateError::DataTooShort {
                expected: required,
                actual: update.data.len(),
            });
        }

        // Java first assigns scale, clears/repopulates decorations, then blits. Keep that
        // sequence for every well-formed packet; excess byte-array data is intentionally
        // ignored because `S34PacketMaps.setMapdataTo` indexes only width * height bytes.
        self.scale = packet.scale;
        self.replace_decorations(&packet.icons);
        for x in 0..width {
            for y in 0..height {
                self.colors[min_x + x + (min_y + y) * MAP_SIZE] = update.data[x + y * width];
            }
        }
        // Any pixel rectangle, even a zero-sized or identical one: a
        // spurious re-upload costs one 64 KiB write, a missed one shows a stale image.
        self.revision = NEXT_MAP_REVISION.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    fn replace_decorations(&mut self, icons: &[s2c::MapIcon]) {
        self.decorations.clear();
        self.decorations
            .extend(icons.iter().enumerate().map(MapDecoration::from));
    }
}

/// A map-pixel rectangle that cannot be represented by the 128 x 128 `MapData.colors` image.
///
/// The protocol decoder intentionally has no such validation because its job is to reproduce
/// the wire shape. This boundary prevents malformed network input from panicking in world state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MapUpdateError {
    /// The rectangle crosses the fixed map-image boundary.
    RectangleOutOfBounds {
        /// Rectangle width from the packet.
        width: u8,
        /// Rectangle height from the packet.
        height: u8,
        /// Rectangle left edge from the packet.
        min_x: u8,
        /// Rectangle top edge from the packet.
        min_y: u8,
    },
    /// The rectangle requests more pixels than the packet's byte array contains.
    DataTooShort {
        /// `width * height` bytes required by Java's indexing loop.
        expected: usize,
        /// Bytes actually present in the packet.
        actual: usize,
    },
}

#[cfg(test)]
mod tests {
    use super::{MapData, MapUpdateError, MAP_SIZE};
    use mc_protocol::play::s2c::{MapIcon, MapPixelUpdate, Maps};

    fn icon(icon_type: u8, x: i8) -> MapIcon {
        MapIcon {
            icon_type,
            x,
            z: -x,
            direction: icon_type,
        }
    }

    #[test]
    fn packet_replaces_scale_and_decorations_even_without_pixels() {
        let mut map = MapData::new(23);
        map.apply_packet(&Maps {
            map_id: 23,
            scale: 1,
            icons: vec![icon(9, 9)],
            update: None,
        })
        .unwrap();
        map.apply_packet(&Maps {
            map_id: 23,
            scale: -7,
            icons: vec![icon(1, -4), icon(2, 7)],
            update: None,
        })
        .unwrap();

        assert_eq!(
            map.scale, -7,
            "setMapdataTo must not clamp the packet scale"
        );
        assert_eq!(map.decorations.len(), 2);
        assert_eq!(map.decorations[0].key, "icon-0");
        assert_eq!(map.decorations[1].key, "icon-1");
        assert_eq!(map.decorations[0].x, -4);
        assert_eq!(map.decorations[1].direction, 2);
    }

    #[test]
    fn packet_blits_a_rectangle_in_java_x_then_y_loop_order() {
        let mut map = MapData::new(4);
        map.colors.fill(0xEE);
        map.apply_packet(&Maps {
            map_id: 4,
            scale: 3,
            icons: vec![],
            update: Some(MapPixelUpdate {
                width: 2,
                height: 3,
                min_x: 5,
                min_y: 7,
                data: vec![10, 11, 12, 13, 14, 15],
            }),
        })
        .unwrap();

        // Java reads `mapData[x + y * width]`, not a transposed `y + x * height` layout.
        assert_eq!(map.colors[5 + 7 * MAP_SIZE], 10);
        assert_eq!(map.colors[6 + 7 * MAP_SIZE], 11);
        assert_eq!(map.colors[5 + 8 * MAP_SIZE], 12);
        assert_eq!(map.colors[6 + 8 * MAP_SIZE], 13);
        assert_eq!(map.colors[5 + 9 * MAP_SIZE], 14);
        assert_eq!(map.colors[6 + 9 * MAP_SIZE], 15);
        assert_eq!(map.colors[4 + 7 * MAP_SIZE], 0xEE);
        assert_eq!(map.colors[5 + 10 * MAP_SIZE], 0xEE);
    }

    #[test]
    fn malformed_rectangle_is_rejected_without_partial_state_mutation() {
        let mut map = MapData::new(8);
        map.apply_packet(&Maps {
            map_id: 8,
            scale: 2,
            icons: vec![icon(3, 2)],
            update: None,
        })
        .unwrap();
        map.colors[0] = 0xAB;
        let before = map.clone();
        let error = map
            .apply_packet(&Maps {
                map_id: 8,
                scale: 4,
                icons: vec![icon(5, 1)],
                update: Some(MapPixelUpdate {
                    width: 2,
                    height: 1,
                    min_x: 127,
                    min_y: 0,
                    data: vec![1, 2],
                }),
            })
            .unwrap_err();

        assert!(matches!(error, MapUpdateError::RectangleOutOfBounds { .. }));
        assert_eq!(
            map, before,
            "validation must finish before Java-shaped mutation begins"
        );
    }

    #[test]
    fn surplus_packet_bytes_do_not_spill_beyond_the_declared_rectangle() {
        let mut map = MapData::new(9);
        map.apply_packet(&Maps {
            map_id: 9,
            scale: 0,
            icons: vec![],
            update: Some(MapPixelUpdate {
                width: 1,
                height: 1,
                min_x: 127,
                min_y: 127,
                data: vec![0x3A, 0xFF],
            }),
        })
        .unwrap();

        assert_eq!(map.colors[MAP_SIZE * MAP_SIZE - 1], 0x3A);
        assert_eq!(map.colors[0], 0, "only width * height bytes are consumed");
    }

    /// The revision moves on every pixel rectangle and only then, is
    /// unique across separate maps (worlds), and stays out of equality.
    #[test]
    fn revision_moves_with_pixels_only_and_never_repeats() {
        let pixel = |value: u8| Maps {
            map_id: 12,
            scale: 0,
            icons: vec![],
            update: Some(MapPixelUpdate {
                width: 1,
                height: 1,
                min_x: 3,
                min_y: 4,
                data: vec![value],
            }),
        };
        let mut map = MapData::new(12);
        assert_eq!(map.revision, 0, "a blank map");
        map.apply_packet(&pixel(40)).unwrap();
        let first = map.revision;
        assert_ne!(first, 0);
        // Decorations alone leave the image, and so the revision, untouched.
        map.apply_packet(&Maps {
            map_id: 12,
            scale: 0,
            icons: vec![icon(1, 2)],
            update: None,
        })
        .unwrap();
        assert_eq!(map.revision, first);
        // A rejected rectangle changes nothing.
        let mut bad = pixel(41);
        bad.update.as_mut().unwrap().min_x = 128;
        assert!(map.apply_packet(&bad).is_err());
        assert_eq!(map.revision, first);
        map.apply_packet(&pixel(41)).unwrap();
        assert!(map.revision > first);

        // Another map with the same id and the same image (a second world) never shares a
        // revision with the first, yet compares equal.
        let mut other = MapData::new(12);
        other.apply_packet(&pixel(40)).unwrap();
        other.apply_packet(&pixel(41)).unwrap();
        other.decorations.clone_from(&map.decorations);
        assert_ne!(other.revision, map.revision);
        assert_eq!(other, map);
    }
}
