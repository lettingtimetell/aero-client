//! Texture atlas bin-packing (java: `Stitcher.java`).
//!
//! Pure geometry: given a list of `(width, height)` tile sizes, decides where each one lands
//! in the final sheet. No pixel data here — [`crate::atlas`] blits pixels once placement is
//! known.
//!
//! Ported close to line-for-line: a greedy guillotine packer that tries each existing slot
//! (and its rotation) before growing the sheet, growing along whichever of width/height keeps
//! the sheet closest to square. The growth heuristics are exactly Java's, since a different
//! (even equally "good") packing order would shuffle sprites to different atlas regions than
//! vanilla — not incorrect, but pointless to diverge from without reason.

use std::cmp::Ordering;

use thiserror::Error;

/// Where one input tile landed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Placement {
    /// Index into the input `sizes` slice this placement is for.
    pub index: usize,
    /// X origin in the final sheet, in pixels.
    pub x: u32,
    /// Y origin in the final sheet, in pixels.
    pub y: u32,
    /// Whether the tile was placed rotated 90 degrees (width/height swapped).
    pub rotated: bool,
}

/// The finished layout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StitchResult {
    /// Final sheet width in pixels.
    pub width: u32,
    /// Final sheet height in pixels.
    pub height: u32,
    /// One entry per input tile, in no particular order.
    pub placements: Vec<Placement>,
}

/// A tile that will not fit within `max_width`x`max_height` no matter how the sheet grows.
#[derive(Debug, Error)]
#[error("unable to fit tile {index} ({width}x{height}) into a {max_width}x{max_height} atlas")]
pub struct StitchError {
    /// Index into the input slice.
    pub index: usize,
    /// The tile's width.
    pub width: u32,
    /// The tile's height.
    pub height: u32,
    /// The atlas's configured maximum width.
    pub max_width: u32,
    /// The atlas's configured maximum height.
    pub max_height: u32,
}

/// An atlas tile together with the icon name Java uses as the final sort key.
///
/// [`stitch`] and [`stitch_with_mipmap`] keep their compact `(width, height)` API for callers
/// that do not have sprite names.  Callers that mirror `TextureAtlasSprite.getIconName()` should
/// use [`stitch_named`] (or [`stitch_with_mipmap_named`]) so equal-sized sprites have the same
/// stable order as vanilla.  `None` has Java's null-name ordering: it sorts before every named
/// sprite and two null names compare equal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NamedSize {
    /// Java's `TextureAtlasSprite.getIconName()`, or `None` for a null icon name.
    pub name: Option<String>,
    /// Source width in pixels.
    pub width: u32,
    /// Source height in pixels.
    pub height: u32,
}

impl NamedSize {
    /// Creates a named tile.
    pub fn new(name: impl Into<String>, width: u32, height: u32) -> Self {
        Self {
            name: Some(name.into()),
            width,
            height,
        }
    }

    /// Creates a tile with Java's null icon name.
    pub fn unnamed(width: u32, height: u32) -> Self {
        Self {
            name: None,
            width,
            height,
        }
    }
}

#[derive(Debug, Clone)]
struct Holder {
    index: usize,
    icon_name: Option<String>,
    width: u32,
    height: u32,
    rotated: bool,
}

impl Holder {
    fn w(&self) -> u32 {
        if self.rotated {
            self.height
        } else {
            self.width
        }
    }
    fn h(&self) -> u32 {
        if self.rotated {
            self.width
        } else {
            self.height
        }
    }
    fn rotate(&mut self) {
        self.rotated = !self.rotated;
    }
}

#[derive(Debug)]
enum Slot {
    Leaf {
        x: u32,
        y: u32,
        w: u32,
        h: u32,
        holder: Option<Holder>,
    },
    Split {
        x: u32,
        y: u32,
        w: u32,
        h: u32,
        children: Vec<Slot>,
    },
}

impl Slot {
    fn leaf(x: u32, y: u32, w: u32, h: u32) -> Slot {
        Slot::Leaf {
            x,
            y,
            w,
            h,
            holder: None,
        }
    }

    fn dims(&self) -> (u32, u32, u32, u32) {
        match self {
            Slot::Leaf { x, y, w, h, .. } | Slot::Split { x, y, w, h, .. } => (*x, *y, *w, *h),
        }
    }

    /// java: `Stitcher.Slot.addSlot` — tries to place `holder` in this slot, splitting it
    /// (guillotine-style) if the holder doesn't exactly fill it.
    fn add(&mut self, holder: &Holder) -> bool {
        let (sx, sy, sw, sh) = self.dims();
        match self {
            Slot::Split { children, .. } => {
                for c in children.iter_mut() {
                    if c.add(holder) {
                        return true;
                    }
                }
                false
            }
            Slot::Leaf {
                holder: occupied, ..
            } => {
                if occupied.is_some() {
                    return false;
                }
                let (iw, ih) = (holder.w(), holder.h());
                if iw > sw || ih > sh {
                    return false;
                }
                if iw == sw && ih == sh {
                    *occupied = Some(holder.clone());
                    return true;
                }
                let mut children = vec![Slot::leaf(sx, sy, iw, ih)];
                let kw = sw - iw;
                let kh = sh - ih;
                if kh > 0 && kw > 0 {
                    let a = sh.max(kw);
                    let b = sw.max(kh);
                    if a >= b {
                        children.push(Slot::leaf(sx, sy + ih, iw, kh));
                        children.push(Slot::leaf(sx + iw, sy, kw, sh));
                    } else {
                        children.push(Slot::leaf(sx + iw, sy, kw, ih));
                        children.push(Slot::leaf(sx, sy + ih, sw, kh));
                    }
                } else if kw == 0 {
                    children.push(Slot::leaf(sx, sy + ih, iw, kh));
                } else if kh == 0 {
                    children.push(Slot::leaf(sx + iw, sy, kw, ih));
                }
                let placed = children[0].add(holder);
                debug_assert!(placed);
                *self = Slot::Split {
                    x: sx,
                    y: sy,
                    w: sw,
                    h: sh,
                    children,
                };
                true
            }
        }
    }

    fn collect(&self, out: &mut Vec<Placement>) {
        match self {
            Slot::Leaf { x, y, holder, .. } => {
                if let Some(h) = holder {
                    out.push(Placement {
                        index: h.index,
                        x: *x,
                        y: *y,
                        rotated: h.rotated,
                    });
                }
            }
            Slot::Split { children, .. } => {
                for c in children {
                    c.collect(out);
                }
            }
        }
    }
}

fn round_up_pow2(v: u32) -> u32 {
    if v == 0 {
        return 0;
    }
    if v == 1 {
        return 1;
    }
    // Java's helper has no representable result above 2^31.  Saturating here keeps malformed
    // custom-pack dimensions from turning a debug-build shift into a panic; growth still checks
    // the unrounded addition, so the sentinel cannot authorize an overflowing atlas.
    v.checked_next_power_of_two().unwrap_or(u32::MAX)
}

/// Packs `sizes` (width, height per index) into a sheet no larger than
/// `max_width`x`max_height`, optionally rounding the final sheet up to a power of two on both
/// axes (java: `Stitcher`, `forcePowerOf2 = true` for the block/item atlas).
///
/// # Errors
/// [`StitchError`] if a tile cannot fit even in an empty, maximally grown sheet.
pub fn stitch(
    sizes: &[(u32, u32)],
    max_width: u32,
    max_height: u32,
    force_pow2: bool,
) -> Result<StitchResult, StitchError> {
    stitch_with_mipmap(sizes, max_width, max_height, force_pow2, 0)
}

/// Packs named tiles with Java's icon-name tie-breaker.
pub fn stitch_named(
    sizes: &[NamedSize],
    max_width: u32,
    max_height: u32,
    force_pow2: bool,
) -> Result<StitchResult, StitchError> {
    stitch_with_mipmap_named(sizes, max_width, max_height, force_pow2, 0)
}

/// As [`stitch`], but reserves each tile on a multiple of `2^mipmap_level`.
///
/// Java's `TextureMap` passes its global mip count into `Stitcher`: a 24px sprite at level 4
/// occupies a 32px slot even though only its first 24 texels are populated at level zero.  The
/// alignment keeps the sprite's separately-generated lower levels on integral atlas texels and
/// prevents neighbouring sprites from bleeding into one another.
pub fn stitch_with_mipmap(
    sizes: &[(u32, u32)],
    max_width: u32,
    max_height: u32,
    force_pow2: bool,
    mipmap_level: u32,
) -> Result<StitchResult, StitchError> {
    let named_sizes: Vec<NamedSize> = sizes
        .iter()
        .map(|&(width, height)| NamedSize::unnamed(width, height))
        .collect();
    stitch_with_mipmap_named(
        &named_sizes,
        max_width,
        max_height,
        force_pow2,
        mipmap_level,
    )
}

/// As [`stitch_named`], but reserves each tile on a multiple of `2^mipmap_level`.
pub fn stitch_with_mipmap_named(
    sizes: &[NamedSize],
    max_width: u32,
    max_height: u32,
    force_pow2: bool,
    mipmap_level: u32,
) -> Result<StitchResult, StitchError> {
    let mut holders: Vec<Holder> = sizes
        .iter()
        .enumerate()
        .map(|(index, size)| Holder {
            index,
            icon_name: size.name.clone(),
            width: mipmap_dimension(size.width, mipmap_level),
            height: mipmap_dimension(size.height, mipmap_level),
            rotated: mipmap_dimension(size.height, mipmap_level)
                > mipmap_dimension(size.width, mipmap_level),
        })
        .collect();

    // java: Arrays.sort / Holder.compareTo — descending height, descending width, then the
    // nullable icon name in ascending order.  Do not use the input index for named holders:
    // custom packs can enumerate the same sprites in a different order, while Java's icon-name
    // comparator keeps their atlas coordinates stable.
    holders.sort_by(|a, b| {
        b.h()
            .cmp(&a.h())
            .then_with(|| b.w().cmp(&a.w()))
            .then_with(|| match (&a.icon_name, &b.icon_name) {
                (None, None) => Ordering::Equal,
                (None, Some(_)) => Ordering::Less,
                (Some(_), None) => Ordering::Greater,
                (Some(a), Some(b)) => a.cmp(b),
            })
    });

    let mut slots: Vec<Slot> = Vec::new();
    let mut cur_w: u32 = 0;
    let mut cur_h: u32 = 0;

    for mut holder in holders {
        if !allocate(
            &mut slots,
            &mut cur_w,
            &mut cur_h,
            max_width,
            max_height,
            force_pow2,
            &mut holder,
        ) {
            return Err(StitchError {
                index: holder.index,
                width: holder.width,
                height: holder.height,
                max_width,
                max_height,
            });
        }
    }

    if force_pow2 {
        cur_w = round_up_pow2(cur_w);
        cur_h = round_up_pow2(cur_h);
    }

    let mut placements = Vec::with_capacity(sizes.len());
    for s in &slots {
        s.collect(&mut placements);
    }
    Ok(StitchResult {
        width: cur_w,
        height: cur_h,
        placements,
    })
}

/// java: `Stitcher.getMipmapDimension`.
fn mipmap_dimension(value: u32, mipmap_level: u32) -> u32 {
    let quantum = 1u32.checked_shl(mipmap_level).unwrap_or(u32::MAX);
    value
        .saturating_add(quantum.saturating_sub(1))
        .saturating_div(quantum)
        .saturating_mul(quantum)
}

fn allocate(
    slots: &mut Vec<Slot>,
    cur_w: &mut u32,
    cur_h: &mut u32,
    max_w: u32,
    max_h: u32,
    force_pow2: bool,
    holder: &mut Holder,
) -> bool {
    for slot in slots.iter_mut() {
        if slot.add(holder) {
            return true;
        }
        holder.rotate();
        if slot.add(holder) {
            return true;
        }
        holder.rotate();
    }
    expand_and_allocate(slots, cur_w, cur_h, max_w, max_h, force_pow2, holder)
}

/// java: `Stitcher.expandAndAllocateSlot`
fn expand_and_allocate(
    slots: &mut Vec<Slot>,
    cur_w: &mut u32,
    cur_h: &mut u32,
    max_w: u32,
    max_h: u32,
    force_pow2: bool,
    holder: &mut Holder,
) -> bool {
    let i = holder.w().min(holder.h());
    let empty = *cur_w == 0 && *cur_h == 0;

    // java: `Stitcher.expandAndAllocateSlot`.  In the forcePowerOf2 branch Java compares
    // pre-rounded current and candidate dimensions, not the raw dimensions.  That distinction
    // matters for custom packs whose running width/height sit just across a power-of-two edge.
    let grow_right = if force_pow2 {
        let current_w = round_up_pow2(*cur_w);
        let current_h = round_up_pow2(*cur_h);
        let candidate_w = cur_w.checked_add(i).map(round_up_pow2);
        let candidate_h = cur_h.checked_add(i).map(round_up_pow2);
        let (Some(candidate_w), Some(candidate_h)) = (candidate_w, candidate_h) else {
            return false;
        };
        let fits_w = candidate_w <= max_w;
        let fits_h = candidate_h <= max_h;
        if !fits_w && !fits_h {
            return false;
        }

        let width_changes = current_w != candidate_w;
        let height_changes = current_h != candidate_h;
        if width_changes ^ height_changes {
            !width_changes
        } else {
            fits_w && current_w <= current_h
        }
    } else {
        let fits_w = cur_w.checked_add(i).is_some_and(|w| w <= max_w);
        let fits_h = cur_h.checked_add(i).is_some_and(|h| h <= max_h);
        if !fits_w && !fits_h {
            return false;
        }
        fits_w && (empty || *cur_w <= *cur_h)
    };

    let j = holder.w().max(holder.h());
    let Some(axis_sum) = (if grow_right { *cur_h } else { *cur_w }).checked_add(j) else {
        return false;
    };
    if round_up_pow2(axis_sum) > if grow_right { max_h } else { max_w } {
        return false;
    }

    if grow_right {
        if holder.w() > holder.h() {
            holder.rotate();
        }
        if *cur_h == 0 {
            *cur_h = holder.h();
        }
        let Some(new_width) = cur_w.checked_add(holder.w()) else {
            return false;
        };
        let mut slot = Slot::leaf(*cur_w, 0, holder.w(), *cur_h);
        *cur_w = new_width;
        let placed = slot.add(holder);
        debug_assert!(placed);
        slots.push(slot);
    } else {
        // Java's grow-down branch does not rotate before constructing its slot.  A non-square
        // custom sprite can therefore be wider than the running sheet (the original Java code
        // silently ignores the failed `addSlot` result).  Keep the atlas total and placement
        // invariants intact by taking the same holder's other orientation when it fits.
        if holder.w() > *cur_w {
            holder.rotate();
            if holder.w() > *cur_w {
                return false;
            }
        }
        let Some(new_height) = cur_h.checked_add(holder.h()) else {
            return false;
        };
        let mut slot = Slot::leaf(0, *cur_h, *cur_w, holder.h());
        *cur_h = new_height;
        let placed = slot.add(holder);
        debug_assert!(placed);
        slots.push(slot);
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packs_uniform_16x16_tiles_without_overlap() {
        let sizes = vec![(16u32, 16u32); 40];
        let r = stitch(&sizes, 1024, 1024, true).unwrap();
        assert_eq!(r.placements.len(), 40);
        assert!(r.width.is_power_of_two());
        assert!(r.height.is_power_of_two());
        assert_no_overlap(&r.placements, &sizes);
    }

    #[test]
    fn packs_mixed_sizes_without_overlap() {
        let sizes = vec![
            (16, 16),
            (32, 32),
            (16, 32),
            (64, 16),
            (16, 16),
            (128, 128),
            (8, 8),
        ];
        let r = stitch(&sizes, 2048, 2048, true).unwrap();
        assert_eq!(r.placements.len(), sizes.len());
        assert_no_overlap(&r.placements, &sizes);
    }

    #[test]
    fn everything_fits_within_the_reported_sheet() {
        let sizes: Vec<(u32, u32)> = (0..100).map(|i| (16, 16 + (i % 3) * 16)).collect();
        let r = stitch(&sizes, 4096, 4096, true).unwrap();
        for p in &r.placements {
            let (w, h) = sizes[p.index];
            let (w, h) = if p.rotated { (h, w) } else { (w, h) };
            assert!(p.x + w <= r.width, "{p:?} exceeds sheet width {}", r.width);
            assert!(
                p.y + h <= r.height,
                "{p:?} exceeds sheet height {}",
                r.height
            );
        }
    }

    #[test]
    fn tile_too_big_for_the_cap_errors_instead_of_panicking() {
        let sizes = vec![(4096, 4096)];
        let err = stitch(&sizes, 1024, 1024, true).unwrap_err();
        assert_eq!(err.index, 0);
    }

    #[test]
    fn power_of_two_rounding_is_applied_to_the_final_sheet_only() {
        let sizes = vec![(16, 16), (16, 16), (16, 16)]; // 48x16 raw if packed in a row
        let r = stitch(&sizes, 1024, 1024, true).unwrap();
        assert!(r.width.is_power_of_two());
        assert!(r.height.is_power_of_two());
    }

    #[test]
    fn mipmap_stitching_rounds_odd_tiles_to_every_level_quantum() {
        // Java's Stitcher.Holder reserves ceil(d / 2^mip) * 2^mip, rather than the
        // source tile's raw extent.  These deliberately awkward dimensions exercise every
        // branch of that rounding at level 3 (an 8px quantum), including a tile taller than
        // it is wide.  Without the reservation, a lower atlas mip could share texels with a
        // neighbouring source tile.
        const MIP_LEVEL: u32 = 3;
        const QUANTUM: u32 = 1 << MIP_LEVEL;
        let sizes = [(13, 19), (25, 10), (9, 33)];
        // ceil((13, 19) / 8) * 8, etc.; keep the Java outputs explicit so this also detects
        // changing the helper and the expectation in lockstep.
        let reserved_sizes = [(16, 24), (32, 16), (16, 40)];
        let layout = stitch_with_mipmap(&sizes, 128, 128, true, MIP_LEVEL).unwrap();

        assert_eq!(layout.placements.len(), sizes.len());
        for placement in &layout.placements {
            let (w, h) = reserved_sizes[placement.index];
            let (w, h) = if placement.rotated { (h, w) } else { (w, h) };
            assert_eq!(placement.x % QUANTUM, 0, "{placement:?}");
            assert_eq!(placement.y % QUANTUM, 0, "{placement:?}");
            assert!(placement.x + w <= layout.width, "{placement:?}");
            assert!(placement.y + h <= layout.height, "{placement:?}");
        }

        // Check the padded holder extents, not the source image extents.  The raw images may
        // be closer together; their invisible reserved tails are what preserve lower mips.
        for i in 0..layout.placements.len() {
            for j in (i + 1)..layout.placements.len() {
                let a = &layout.placements[i];
                let b = &layout.placements[j];
                let padded_dims = |placement: &Placement| {
                    let (w, h) = reserved_sizes[placement.index];
                    if placement.rotated {
                        (h, w)
                    } else {
                        (w, h)
                    }
                };
                let (aw, ah) = padded_dims(a);
                let (bw, bh) = padded_dims(b);
                let overlap_x = a.x < b.x + bw && b.x < a.x + aw;
                let overlap_y = a.y < b.y + bh && b.y < a.y + ah;
                assert!(!(overlap_x && overlap_y), "{a:?} overlaps {b:?}");
            }
        }
    }

    #[test]
    fn equal_dimension_sprites_use_icon_name_order() {
        let sizes = [
            NamedSize::new("zulu", 16, 16),
            NamedSize::new("alpha", 16, 16),
        ];
        let layout = stitch_named(&sizes, 64, 64, true).unwrap();

        let alpha = layout
            .placements
            .iter()
            .find(|placement| sizes[placement.index].name.as_deref() == Some("alpha"))
            .unwrap();
        let zulu = layout
            .placements
            .iter()
            .find(|placement| sizes[placement.index].name.as_deref() == Some("zulu"))
            .unwrap();
        assert_eq!((alpha.x, alpha.y), (0, 0));
        assert_eq!((zulu.x, zulu.y), (16, 0));
    }

    #[test]
    fn named_packing_is_independent_of_input_permutation() {
        let first = [
            NamedSize::new("zulu", 16, 16),
            NamedSize::new("alpha", 16, 16),
            NamedSize::new("middle", 8, 4),
            NamedSize::new("bravo", 4, 8),
        ];
        let second = [
            first[3].clone(),
            first[0].clone(),
            first[2].clone(),
            first[1].clone(),
        ];
        let first_layout = stitch_named(&first, 64, 64, true).unwrap();
        let second_layout = stitch_named(&second, 64, 64, true).unwrap();

        assert_eq!(first_layout.width, second_layout.width);
        assert_eq!(first_layout.height, second_layout.height);
        assert_eq!(
            named_coordinates(&first, &first_layout),
            named_coordinates(&second, &second_layout)
        );
    }

    #[test]
    fn force_power_of_two_growth_uses_rounded_candidates() {
        // After the 4x4 and first 1x4 holders, the raw dimensions are 5x4.  Java sees rounded
        // current dimensions 8x4 and rounded candidates 8x8, so it grows right.  A raw-size
        // heuristic would grow down instead and finish at 8x8 rather than Java's 8x4.
        let sizes = [
            NamedSize::new("base", 4, 4),
            NamedSize::new("left", 1, 4),
            NamedSize::new("right", 1, 4),
        ];
        let layout = stitch_named(&sizes, 8, 8, true).unwrap();

        assert_eq!((layout.width, layout.height), (8, 4));
        let left = layout
            .placements
            .iter()
            .find(|placement| sizes[placement.index].name.as_deref() == Some("left"))
            .unwrap();
        let right = layout
            .placements
            .iter()
            .find(|placement| sizes[placement.index].name.as_deref() == Some("right"))
            .unwrap();
        assert_eq!((left.x, left.y), (4, 0));
        assert_eq!((right.x, right.y), (5, 0));
    }

    #[test]
    fn rotation_is_preserved_when_a_holder_fits_a_free_slot() {
        let sizes = [
            NamedSize::new("base", 16, 16),
            NamedSize::new("alpha", 8, 4),
            NamedSize::new("bravo", 4, 8),
        ];
        let layout = stitch_named(&sizes, 64, 64, true).unwrap();
        let bravo = layout
            .placements
            .iter()
            .find(|placement| sizes[placement.index].name.as_deref() == Some("bravo"))
            .unwrap();
        let alpha = layout
            .placements
            .iter()
            .find(|placement| sizes[placement.index].name.as_deref() == Some("alpha"))
            .unwrap();
        assert_eq!((alpha.x, alpha.y), (16, 0));
        assert!(alpha.rotated);
        assert_eq!((bravo.x, bravo.y), (16, 8));
        assert!(!bravo.rotated);
    }

    #[test]
    fn ordinary_vanilla_compatible_pack_has_stable_golden_layout() {
        let sizes = [
            NamedSize::new("stone", 16, 16),
            NamedSize::new("planks", 32, 16),
            NamedSize::new("glass", 8, 8),
            NamedSize::new("dirt", 16, 32),
        ];
        let layout = stitch_named(&sizes, 128, 128, true).unwrap();
        let mut actual = named_coordinates(&sizes, &layout);
        actual.sort_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(
            actual,
            vec![
                ("dirt".to_owned(), 0, 0, false),
                ("glass".to_owned(), 32, 16, false),
                ("planks".to_owned(), 16, 0, true),
                ("stone".to_owned(), 32, 0, false),
            ]
        );
        assert_eq!((layout.width, layout.height), (64, 32));
    }

    fn named_coordinates(
        sizes: &[NamedSize],
        layout: &StitchResult,
    ) -> Vec<(String, u32, u32, bool)> {
        let mut coordinates = layout
            .placements
            .iter()
            .map(|placement| {
                (
                    sizes[placement.index].name.clone().unwrap(),
                    placement.x,
                    placement.y,
                    placement.rotated,
                )
            })
            .collect::<Vec<_>>();
        coordinates.sort_by(|a, b| a.0.cmp(&b.0));
        coordinates
    }

    fn assert_no_overlap(placements: &[Placement], sizes: &[(u32, u32)]) {
        for i in 0..placements.len() {
            for j in (i + 1)..placements.len() {
                let a = &placements[i];
                let b = &placements[j];
                let (aw, ah) = dims_of(a, sizes);
                let (bw, bh) = dims_of(b, sizes);
                let overlap_x = a.x < b.x + bw && b.x < a.x + aw;
                let overlap_y = a.y < b.y + bh && b.y < a.y + ah;
                assert!(!(overlap_x && overlap_y), "{a:?} overlaps {b:?}");
            }
        }
    }

    fn dims_of(p: &Placement, sizes: &[(u32, u32)]) -> (u32, u32) {
        let (w, h) = sizes[p.index];
        if p.rotated {
            (h, w)
        } else {
            (w, h)
        }
    }
}
