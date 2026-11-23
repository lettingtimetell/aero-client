//! Item-frame map art: every filled map shares a few atlas pages ("map item displays extremely laggy, large groups of them making artwork feel
//! terrible").
//!
//! # What was slow
//!
//! Every visible framed map, every frame, converted its 16 384 palette bytes, made two 64 KiB
//! heap copies, created a **new** GPU texture, view and bind group, uploaded 64 KiB, and drew as
//! its own batch between two item-frame shell batches. A 384-map wall therefore created 384
//! textures and uploaded 24 MiB per frame, and issued ~770 draws with a pipeline and bind-group
//! switch each.
//!
//! # What this does instead
//!
//! * **Upload on change only.** A map's image is written once, and again only when its pixels
//!   change: the caller passes `mc_world::MapData::revision`, a process-wide stamp bumped by
//!   every applied S34 pixel rectangle. That is vanilla's own cadence:
//!   `MapItemRenderer.Instance.updateMapTexture` runs from the S34 handler, never per frame.
//! * **One texture for many maps.** Maps live in [`PAGE_SIDE`]-square RGBA8 pages of
//!   [`SLOTS_PER_PAGE`] slots, so consecutive map quads share a texture and merge into one draw.
//!   Pages are created lazily, on the first framed map in view: no maps, no page, no cost.
//! * **The gutter is vanilla's wrap.** A slot is [`SLOT_SIDE`] texels square: the 128x128 image
//!   plus a one-texel border holding the *opposite* edge. Vanilla's `DynamicTexture` samples
//!   with `GL_REPEAT` and `GL_NEAREST`, so a fragment whose interpolated UV lands a hair past the
//!   quad's edge reads the far edge of the same map; the gutter returns exactly that texel, and
//!   never a neighbouring map's.
//! * **Bounded.** At most [`MAX_PAGES`] pages ([`MAX_PAGES`] x 4 MiB of VRAM, 784 maps). Beyond
//!   that the least recently drawn map gives up its slot; a map drawn this frame never does.
//!
//! The slot bookkeeping here is GPU-free; [`super::render::EntityPhase::map_atlas_slot`] does the
//! texture work.

use std::collections::HashMap;

/// A vanilla map image's side, in texels.
pub const MAP_SIDE: u32 = 128;
/// A slot's side: the image plus a one-texel wrap gutter on every edge.
pub const SLOT_SIDE: u32 = MAP_SIDE + 2;
/// An atlas page's side, in texels (4 MiB of RGBA8).
pub const PAGE_SIDE: u32 = 1024;
/// Slots across one page row.
pub const SLOTS_PER_ROW: u32 = PAGE_SIDE / SLOT_SIDE;
/// Slots on one page.
pub const SLOTS_PER_PAGE: u32 = SLOTS_PER_ROW * SLOTS_PER_ROW;
/// The most pages the atlas creates before it starts evicting.
pub const MAX_PAGES: usize = 16;

// The slots, gutters included, fit on a page.
const _: () = assert!(SLOTS_PER_ROW * SLOT_SIDE <= PAGE_SIDE);

/// The entity-phase texture key of each page. Static strings, so a draw can name its page while
/// the phase is borrowed mutably.
pub const PAGE_KEYS: [&str; MAX_PAGES] = [
    "mc-render/entity/dynamic/map-atlas/0",
    "mc-render/entity/dynamic/map-atlas/1",
    "mc-render/entity/dynamic/map-atlas/2",
    "mc-render/entity/dynamic/map-atlas/3",
    "mc-render/entity/dynamic/map-atlas/4",
    "mc-render/entity/dynamic/map-atlas/5",
    "mc-render/entity/dynamic/map-atlas/6",
    "mc-render/entity/dynamic/map-atlas/7",
    "mc-render/entity/dynamic/map-atlas/8",
    "mc-render/entity/dynamic/map-atlas/9",
    "mc-render/entity/dynamic/map-atlas/10",
    "mc-render/entity/dynamic/map-atlas/11",
    "mc-render/entity/dynamic/map-atlas/12",
    "mc-render/entity/dynamic/map-atlas/13",
    "mc-render/entity/dynamic/map-atlas/14",
    "mc-render/entity/dynamic/map-atlas/15",
];

/// Where one map's image lives.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MapSlot {
    /// Page index (into [`PAGE_KEYS`]).
    pub page: usize,
    /// Slot index on that page, row-major.
    pub index: u32,
}

impl MapSlot {
    /// The slot's top-left texel on its page (the gutter's corner).
    #[must_use]
    pub const fn origin(self) -> (u32, u32) {
        (
            (self.index % SLOTS_PER_ROW) * SLOT_SIDE,
            (self.index / SLOTS_PER_ROW) * SLOT_SIDE,
        )
    }

    /// The page texture key this slot's quads are drawn with.
    #[must_use]
    pub const fn texture_key(self) -> &'static str {
        PAGE_KEYS[self.page]
    }

    /// Map an image-space UV (`0..=1` across the 128x128 image, as `MapItemRenderer` emits it)
    /// into page UV. Exact at the corners: every term is an integer over a power of two.
    #[must_use]
    pub fn uv(self, uv: [f32; 2]) -> [f32; 2] {
        let (x, y) = self.origin();
        let side = MAP_SIDE as f32;
        let page = PAGE_SIDE as f32;
        [
            (x as f32 + 1.0 + uv[0] * side) / page,
            (y as f32 + 1.0 + uv[1] * side) / page,
        ]
    }
}

/// A resident map image: where it is, and whether every texel is opaque.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MapImage {
    /// Where the image lives.
    pub slot: MapSlot,
    /// Every texel has alpha 255 (a fully explored map). Only such an image may be drawn out of
    /// order: under vanilla's `(ONE, ONE_MINUS_SRC_ALPHA)` map blend it then ignores what is
    /// behind it. Unexplored texels are Java's 16/24-alpha checkerboard.
    pub opaque: bool,
}

/// What [`MapAtlas::plan`] wants the caller to do for one map this frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Plan {
    /// The atlas already holds this revision: draw from the slot.
    Ready(MapImage),
    /// Nothing to draw from the atlas: an image that has never converted, or every slot is in
    /// use by a map drawn this frame. The caller draws its fallback.
    Unavailable,
    /// Write the image into `slot` (creating page `slot.page` first when `new_page`), then
    /// record it with [`MapAtlas::uploaded`] and draw. `previous`: the slot already holds an older
    /// image of this same map, which stays (vanilla keeps its last uploaded `DynamicTexture`)
    /// if the new image cannot be converted.
    Upload {
        /// Destination.
        slot: MapSlot,
        /// The page does not exist yet.
        new_page: bool,
        /// See the variant docs.
        previous: Option<MapImage>,
    },
}

/// Cumulative counters, for tests and the benchmark report.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MapAtlasStats {
    /// Pages created and still alive.
    pub pages: usize,
    /// Maps currently holding a slot.
    pub resident: usize,
    /// Slot uploads since the atlas was created or cleared.
    pub uploads: u64,
    /// Bytes written by those uploads.
    pub upload_bytes: u64,
    /// Maps that lost their slot to a more recently drawn one.
    pub evictions: u64,
}

#[derive(Debug, Clone, Copy)]
struct Resident {
    /// `None`: the map's image has never converted (an invalid palette byte); kept so a broken
    /// map is not re-converted every frame until its revision changes.
    slot: Option<MapSlot>,
    revision: u64,
    last_used: u64,
    /// See [`MapImage::opaque`]; meaningful once the slot holds an image.
    opaque: bool,
}

impl Resident {
    fn image(self) -> Option<MapImage> {
        self.slot.map(|slot| MapImage {
            slot,
            opaque: self.opaque,
        })
    }
}

/// The slot bookkeeping of the shared map atlas.
#[derive(Debug, Default)]
pub struct MapAtlas {
    resident: HashMap<i32, Resident>,
    free: Vec<MapSlot>,
    pages: usize,
    frame: u64,
    uploads: u64,
    upload_bytes: u64,
    evictions: u64,
    /// The gutter-expanded image of the slot being written, reused between uploads.
    staging: Vec<u8>,
}

impl MapAtlas {
    /// Start a frame. Maps drawn from now on are "used this frame" and are never evicted until
    /// the next call.
    pub fn begin_frame(&mut self) {
        self.frame = self.frame.wrapping_add(1);
    }

    /// Decide where `map_id` at `revision` is drawn from, allocating (or evicting) a slot when
    /// the atlas does not hold that revision yet.
    pub fn plan(&mut self, map_id: i32, revision: u64) -> Plan {
        let frame = self.frame;
        if let Some(resident) = self.resident.get_mut(&map_id) {
            resident.last_used = frame;
            if resident.revision == revision {
                return resident.image().map_or(Plan::Unavailable, Plan::Ready);
            }
            resident.revision = revision;
            if let Some(previous) = resident.image() {
                return Plan::Upload {
                    slot: previous.slot,
                    new_page: false,
                    previous: Some(previous),
                };
            }
        }
        let Some((slot, new_page)) = self.allocate() else {
            // Every slot is drawn this frame. Forget the revision so the next frame retries.
            self.resident.remove(&map_id);
            return Plan::Unavailable;
        };
        self.resident.insert(
            map_id,
            Resident {
                slot: Some(slot),
                revision,
                last_used: frame,
                opaque: false,
            },
        );
        Plan::Upload {
            slot,
            new_page,
            previous: None,
        }
    }

    /// Record that the image planned by [`Self::plan`] is now in its slot.
    pub fn uploaded(&mut self, map_id: i32, opaque: bool) -> Option<MapImage> {
        let resident = self.resident.get_mut(&map_id)?;
        resident.opaque = opaque;
        resident.image()
    }

    /// The image for a fresh slot could not be converted: give the slot back and remember the
    /// revision as broken.
    pub fn fail(&mut self, map_id: i32) {
        if let Some(resident) = self.resident.get_mut(&map_id) {
            if let Some(slot) = resident.slot.take() {
                self.free.push(slot);
            }
        }
    }

    /// Expand a 128x128 RGBA8 image into the reused staging buffer with its wrap gutter and
    /// count the upload. Returns whether every texel is opaque ([`MapImage::opaque`]); the
    /// [`SLOT_SIDE`]-square pixels are then in [`Self::take_staging`].
    ///
    /// # Panics
    /// If `rgba` is not exactly 128 x 128 x 4 bytes.
    pub fn stage(&mut self, rgba: &[u8]) -> bool {
        assert_eq!(
            rgba.len(),
            (MAP_SIDE * MAP_SIDE * 4) as usize,
            "map image size"
        );
        let opaque = rgba.chunks_exact(4).all(|texel| texel[3] == u8::MAX);
        let side = SLOT_SIDE as usize;
        let map = MAP_SIDE as usize;
        self.staging.resize(side * side * 4, 0);
        for gy in 0..side {
            // Gutter row 0 is image row 127 and gutter row 129 is image row 0: GL_REPEAT.
            let sy = (gy + map - 1) % map;
            let src_row = &rgba[sy * map * 4..][..map * 4];
            let dst_row = &mut self.staging[gy * side * 4..][..side * 4];
            dst_row[..4].copy_from_slice(&src_row[(map - 1) * 4..]);
            dst_row[4..(map + 1) * 4].copy_from_slice(src_row);
            dst_row[(map + 1) * 4..].copy_from_slice(&src_row[..4]);
        }
        self.uploads += 1;
        self.upload_bytes += self.staging.len() as u64;
        opaque
    }

    /// Take the staging buffer out (for an upload API that needs it by value); give it back
    /// with [`Self::restore_staging`].
    pub fn take_staging(&mut self) -> Vec<u8> {
        std::mem::take(&mut self.staging)
    }

    /// Return the buffer taken by [`Self::take_staging`].
    pub fn restore_staging(&mut self, staging: Vec<u8>) {
        self.staging = staging;
    }

    /// Forget every map and page; returns how many pages existed (the caller frees their
    /// textures). The frame counter keeps running.
    pub fn clear(&mut self) -> usize {
        let pages = self.pages;
        self.resident.clear();
        self.free.clear();
        self.pages = 0;
        self.uploads = 0;
        self.upload_bytes = 0;
        self.evictions = 0;
        self.staging = Vec::new();
        pages
    }

    /// Whether any page exists.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.pages == 0
    }

    /// Current counters.
    #[must_use]
    pub fn stats(&self) -> MapAtlasStats {
        MapAtlasStats {
            pages: self.pages,
            resident: self
                .resident
                .values()
                .filter(|resident| resident.slot.is_some())
                .count(),
            uploads: self.uploads,
            upload_bytes: self.upload_bytes,
            evictions: self.evictions,
        }
    }

    fn allocate(&mut self) -> Option<(MapSlot, bool)> {
        if let Some(slot) = self.free.pop() {
            return Some((slot, false));
        }
        if self.pages < MAX_PAGES {
            let page = self.pages;
            self.pages += 1;
            // Reverse, so the next pops hand out slots 1, 2, 3, ... in order.
            for index in (1..SLOTS_PER_PAGE).rev() {
                self.free.push(MapSlot { page, index });
            }
            return Some((MapSlot { page, index: 0 }, true));
        }
        // Full: the least recently drawn map that was not drawn this frame gives up its slot.
        let frame = self.frame;
        let victim = self
            .resident
            .iter()
            .filter(|(_, resident)| resident.slot.is_some() && resident.last_used != frame)
            .min_by_key(|(id, resident)| (resident.last_used, **id))
            .map(|(id, _)| *id)?;
        let slot = self.resident.remove(&victim)?.slot?;
        self.evictions += 1;
        Some((slot, false))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn page_geometry_fits_and_keys_are_distinct() {
        assert_eq!(SLOTS_PER_ROW, 7);
        assert_eq!(SLOTS_PER_PAGE, 49);
        let mut keys = PAGE_KEYS.to_vec();
        keys.sort_unstable();
        keys.dedup();
        assert_eq!(keys.len(), MAX_PAGES);
        for (page, key) in PAGE_KEYS.iter().enumerate() {
            assert!(key.ends_with(&format!("/{page}")));
        }
    }

    #[test]
    fn uv_corners_land_exactly_on_the_image_inside_the_gutter() {
        let slot = MapSlot { page: 0, index: 8 }; // row 1, column 1
        assert_eq!(slot.origin(), (130, 130));
        assert_eq!(slot.uv([0.0, 0.0]), [131.0 / 1024.0, 131.0 / 1024.0]);
        assert_eq!(slot.uv([1.0, 1.0]), [259.0 / 1024.0, 259.0 / 1024.0]);
        // Texel centres map one to one: image texel 5 is page texel 131 + 5.
        let centre = slot.uv([5.5 / 128.0, 0.5 / 128.0]);
        assert_eq!((centre[0] * 1024.0).floor(), 136.0);
        assert_eq!((centre[1] * 1024.0).floor(), 131.0);
        let last = MapSlot {
            page: 3,
            index: SLOTS_PER_PAGE - 1,
        };
        assert_eq!(last.origin(), (780, 780));
        assert_eq!(last.texture_key(), "mc-render/entity/dynamic/map-atlas/3");
    }

    #[test]
    fn a_map_uploads_once_and_again_only_when_its_revision_moves() {
        let mut atlas = MapAtlas::default();
        atlas.begin_frame();
        let Plan::Upload {
            slot,
            new_page: true,
            previous: None,
        } = atlas.plan(7, 1)
        else {
            panic!("the first map creates page 0");
        };
        assert_eq!(slot, MapSlot { page: 0, index: 0 });
        let image = atlas.uploaded(7, true).unwrap();
        assert_eq!(image, MapImage { slot, opaque: true });
        for _ in 0..100 {
            atlas.begin_frame();
            assert_eq!(atlas.plan(7, 1), Plan::Ready(image));
        }
        atlas.begin_frame();
        assert_eq!(
            atlas.plan(7, 2),
            Plan::Upload {
                slot,
                new_page: false,
                previous: Some(image),
            },
            "a new revision re-uses the map's own slot"
        );
        // The new image has an unexplored corner: it is no longer opaque.
        let partial = atlas.uploaded(7, false).unwrap();
        assert_eq!(atlas.plan(7, 2), Plan::Ready(partial));
        assert!(!partial.opaque);
        assert_eq!(partial.slot, slot);
    }

    #[test]
    fn maps_fill_a_page_in_order_then_open_the_next() {
        let mut atlas = MapAtlas::default();
        atlas.begin_frame();
        for id in 0..SLOTS_PER_PAGE as i32 {
            let Plan::Upload { slot, new_page, .. } = atlas.plan(id, 1) else {
                panic!("map {id} gets a slot");
            };
            assert_eq!(
                slot,
                MapSlot {
                    page: 0,
                    index: id as u32
                }
            );
            assert_eq!(new_page, id == 0);
        }
        let Plan::Upload {
            slot,
            new_page: true,
            ..
        } = atlas.plan(1000, 1)
        else {
            panic!("the 50th map opens page 1");
        };
        assert_eq!(slot, MapSlot { page: 1, index: 0 });
        assert_eq!(atlas.stats().pages, 2);
        assert_eq!(atlas.stats().resident, 50);
    }

    #[test]
    fn a_broken_image_frees_a_fresh_slot_and_is_not_retried_until_it_changes() {
        let mut atlas = MapAtlas::default();
        atlas.begin_frame();
        let Plan::Upload { slot, .. } = atlas.plan(3, 9) else {
            panic!()
        };
        atlas.fail(3);
        assert_eq!(
            atlas.plan(3, 9),
            Plan::Unavailable,
            "no retry at the same revision"
        );
        assert_eq!(atlas.stats().resident, 0);
        // The freed slot goes to the next map.
        let Plan::Upload { slot: next, .. } = atlas.plan(4, 1) else {
            panic!()
        };
        assert_eq!(next, slot);
        // A fixed image (new revision) gets a slot again.
        assert!(matches!(
            atlas.plan(3, 10),
            Plan::Upload { previous: None, .. }
        ));
    }

    #[test]
    fn a_full_atlas_evicts_the_least_recently_drawn_map_but_never_one_drawn_this_frame() {
        let mut atlas = MapAtlas::default();
        let capacity = MAX_PAGES * SLOTS_PER_PAGE as usize;
        atlas.begin_frame();
        for id in 0..capacity as i32 {
            assert!(matches!(atlas.plan(id, 1), Plan::Upload { .. }));
        }
        // Next frame: draw everything except map 17, which becomes the oldest.
        atlas.begin_frame();
        for id in (0..capacity as i32).filter(|&id| id != 17) {
            assert!(matches!(atlas.plan(id, 1), Plan::Ready(_)));
        }
        let Plan::Upload {
            new_page: false,
            previous: None,
            ..
        } = atlas.plan(-1, 1)
        else {
            panic!("a full atlas evicts");
        };
        assert_eq!(atlas.stats().evictions, 1);
        // Map 17 lost its slot; every other map drawn this frame kept its own.
        assert!(matches!(atlas.plan(17, 1), Plan::Unavailable));
        assert_eq!(atlas.stats().pages, MAX_PAGES);
    }

    #[test]
    fn the_gutter_repeats_the_opposite_edges_like_gl_repeat() {
        let mut atlas = MapAtlas::default();
        let mut image = vec![0u8; (MAP_SIDE * MAP_SIDE * 4) as usize];
        for y in 0..MAP_SIDE {
            for x in 0..MAP_SIDE {
                let i = ((y * MAP_SIDE + x) * 4) as usize;
                image[i..i + 4].copy_from_slice(&[x as u8, y as u8, 7, 255]);
            }
        }
        assert!(atlas.stage(&image), "every texel has alpha 255");
        let staged = atlas.take_staging();
        let side = SLOT_SIDE as usize;
        let texel = |x: usize, y: usize| -> [u8; 4] {
            let i = (y * side + x) * 4;
            staged[i..i + 4].try_into().unwrap()
        };
        assert_eq!(staged.len(), side * side * 4);
        assert_eq!(texel(1, 1), [0, 0, 7, 255], "image texel (0, 0)");
        assert_eq!(
            texel(128, 128),
            [127, 127, 7, 255],
            "image texel (127, 127)"
        );
        assert_eq!(texel(0, 1), [127, 0, 7, 255], "left gutter = right edge");
        assert_eq!(texel(129, 5), [0, 4, 7, 255], "right gutter = left edge");
        assert_eq!(texel(9, 0), [8, 127, 7, 255], "top gutter = bottom edge");
        assert_eq!(texel(9, 129), [8, 0, 7, 255], "bottom gutter = top edge");
        assert_eq!(texel(0, 0), [127, 127, 7, 255], "corner wraps on both axes");
        assert_eq!(texel(129, 129), [0, 0, 7, 255]);
        let stats = atlas.stats();
        assert_eq!(stats.uploads, 1);
        assert_eq!(stats.upload_bytes, (side * side * 4) as u64);

        // One unexplored texel (Java's 16/24-alpha checkerboard) makes the image translucent.
        atlas.restore_staging(staged);
        image[(70 * MAP_SIDE as usize + 3) * 4 + 3] = 24;
        assert!(!atlas.stage(&image));
    }

    #[test]
    fn clear_forgets_maps_and_pages() {
        let mut atlas = MapAtlas::default();
        atlas.begin_frame();
        let _ = atlas.plan(1, 1);
        assert!(!atlas.is_empty());
        assert_eq!(atlas.clear(), 1);
        assert!(atlas.is_empty());
        assert!(matches!(
            atlas.plan(1, 1),
            Plan::Upload { new_page: true, .. }
        ));
    }
}
