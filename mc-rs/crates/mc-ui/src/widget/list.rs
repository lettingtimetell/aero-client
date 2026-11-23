//! Scrolling lists and their scrollbar.
//!
//! java: `client/gui/GuiSlot.java`, `client/gui/GuiListExtended.java`
//!
//! `GuiSlot` is where vanilla's server list, resource-pack list, language list and key-binding
//! list all come from. The parts worth porting exactly are the scroll clamp, the
//! screen-coordinate → slot-index mapping, and the scrollbar thumb geometry; the artwork is a
//! tiled dirt background plus two gradient fades, which [`ScrollList::draw_frame`] reproduces.
//!
//! Entry *contents* are the caller's business: this type owns the viewport and the scroll
//! state, and hands back the rectangle each visible entry should draw into.

use mc_math::math_helper::{clamp_f32, clamp_i32};

use crate::widget::draw::{DrawList, Rect, UiTexture, OPTIONS_BACKGROUND};

/// Vanilla's list content width.
// java: GuiSlot.java:442 (getListWidth)
pub const DEFAULT_LIST_WIDTH: i32 = 220;
/// The scrollbar is 6 pixels wide.
pub const SCROLLBAR_WIDTH: i32 = 6;
/// The thumb is never shorter than this.
pub const MIN_THUMB_HEIGHT: i32 = 32;
/// `GuiSlot.getScrollBarX()`'s offset from the centre of the viewport.
// java: GuiSlot.java:489
pub const DEFAULT_SCROLLBAR_OFFSET: i32 = 124;

/// Where a list puts its scrollbar.
///
/// `GuiSlot.getScrollBarX()` is `protected` precisely so that subclasses can move it, and the
/// three that ship with 1.8.9 move it in two different *shapes* — two nudge the centred default
/// outwards, one measures in from the list's own right edge. An `i32` offset alone cannot
/// express the second, because that list is positioned with `setSlotXBoundsFromLeft` and its
/// bar has to track `right`.
///
/// | Java | Variant |
/// |---|---|
/// | `GuiSlot.getScrollBarX()` | `Centered { offset: 0 }` (the default) |
/// | `ServerSelectionList.java:91` — `super + 30` | `Centered { offset: 30 }` |
/// | `GuiKeyBindingList.java:63` — `super + 15` | `Centered { offset: 15 }` |
/// | `GuiResourcePackList.java:60` — `this.right - 6` | `FromRight { inset: 6 }` |
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScrollbarX {
    /// `width / 2 + 124 + offset`, vanilla's default placement plus a subclass nudge.
    Centered {
        /// Pixels further right than `GuiSlot`'s own `width / 2 + 124`.
        offset: i32,
    },
    /// `right - inset`, measured in from the list's right edge.
    FromRight {
        /// Pixels in from [`ScrollList::right`].
        inset: i32,
    },
    /// A literal GUI-pixel column, for a layout vanilla does not have a rule for.
    Fixed(i32),
}

impl Default for ScrollbarX {
    /// `GuiSlot`'s own placement, so a list that never touches this behaves exactly as before.
    fn default() -> Self {
        ScrollbarX::Centered { offset: 0 }
    }
}

/// The scrollbar's on-screen geometry for one frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Scrollbar {
    /// Left edge of the 6-pixel track.
    pub x: i32,
    /// Top of the track (the viewport top).
    pub track_top: i32,
    /// Bottom of the track.
    pub track_bottom: i32,
    /// Top of the thumb.
    pub thumb_top: i32,
    /// Thumb height, at least [`MIN_THUMB_HEIGHT`].
    pub thumb_height: i32,
}

/// A vertically scrolling list of fixed-height entries.
#[derive(Debug, Clone, PartialEq)]
pub struct ScrollList {
    /// Viewport width. The list content is centred inside it.
    pub width: i32,
    /// Screen height, used only by the top/bottom overlay bands.
    pub height: i32,
    /// Top of the scrolling viewport.
    pub top: i32,
    /// Bottom of the scrolling viewport.
    pub bottom: i32,
    /// Left edge of the viewport.
    pub left: i32,
    /// Right edge of the viewport.
    pub right: i32,
    /// Height of one entry, including its 4-pixel gap.
    pub slot_height: i32,
    /// How many entries there are.
    pub size: usize,
    /// Extra space above the first entry, for a header row.
    pub header_padding: i32,
    /// Whether a header row is drawn.
    pub has_header: bool,
    /// Whether the selected entry gets an outline.
    pub show_selection_box: bool,
    /// The selected entry, or `-1`.
    pub selected: i32,
    /// Content width; entries are centred on the viewport at this width.
    pub list_width: i32,
    /// Where the scrollbar sits — `GuiSlot.getScrollBarX()`'s overridable half.
    ///
    /// Leave it at [`ScrollbarX::default`] for `GuiSlot`'s own geometry. Set it to
    /// `Centered { offset: 30 }` together with `list_width = 305` for `ServerSelectionList`,
    /// which is what makes [`slot_index_at`](Self::slot_index_at) accept the rightmost 28 pixel
    /// columns of a server row instead of rejecting them.
    pub scrollbar_placement: ScrollbarX,
    /// A disabled list still draws but does not scroll on drag.
    pub enabled: bool,
    amount_scrolled: f32,
    initial_click_y: i32,
    scroll_multiplier: f32,
}

impl ScrollList {
    /// A list occupying `top..bottom` of a `width`×`height` screen.
    // java: GuiSlot.java:64
    #[must_use]
    pub fn new(width: i32, height: i32, top: i32, bottom: i32, slot_height: i32) -> Self {
        Self {
            width,
            height,
            top,
            bottom,
            left: 0,
            right: width,
            slot_height,
            size: 0,
            header_padding: 0,
            has_header: false,
            show_selection_box: true,
            selected: -1,
            list_width: DEFAULT_LIST_WIDTH,
            scrollbar_placement: ScrollbarX::default(),
            enabled: true,
            amount_scrolled: 0.0,
            initial_click_y: -2,
            scroll_multiplier: 1.0,
        }
    }

    /// Re-lay the viewport after a resize.
    // java: GuiSlot.java:76 (setDimensions)
    pub fn set_dimensions(&mut self, width: i32, height: i32, top: i32, bottom: i32) {
        self.width = width;
        self.height = height;
        self.top = top;
        self.bottom = bottom;
        self.left = 0;
        self.right = width;
    }

    /// Set the header height. Passing `false` zeroes the padding, as vanilla does.
    // java: GuiSlot.java:95 (setHasListHeader)
    pub fn set_header(&mut self, has_header: bool, padding: i32) {
        self.has_header = has_header;
        self.header_padding = if has_header { padding } else { 0 };
    }

    /// Total scrollable content height.
    // java: GuiSlot.java:121 (getContentHeight)
    #[must_use]
    pub fn content_height(&self) -> i32 {
        self.size as i32 * self.slot_height + self.header_padding
    }

    /// How far the list can scroll. Zero when everything fits.
    ///
    /// Note the `- 4`: vanilla treats the viewport as four pixels shorter than it is, which is
    /// why a list that *just* fits still scrolls a few pixels.
    // java: GuiSlot.java:175 (func_148135_f)
    #[must_use]
    pub fn max_scroll(&self) -> i32 {
        (self.content_height() - (self.bottom - self.top - 4)).max(0)
    }

    /// Current scroll offset in pixels.
    #[must_use]
    pub fn amount_scrolled(&self) -> f32 {
        self.amount_scrolled
    }

    /// Current scroll offset truncated to an integer, which is what all the layout maths uses.
    // java: GuiSlot.java:183 (getAmountScrolled)
    #[must_use]
    pub fn scroll_pixels(&self) -> i32 {
        self.amount_scrolled as i32
    }

    /// Clamp the scroll offset into range.
    // java: GuiSlot.java:170 (bindAmountScrolled)
    pub fn clamp_scroll(&mut self) {
        self.amount_scrolled = clamp_f32(self.amount_scrolled, 0.0, self.max_scroll() as f32);
    }

    /// Scroll by `amount` pixels and clamp. Positive scrolls down.
    // java: GuiSlot.java:196 (scrollBy)
    pub fn scroll_by(&mut self, amount: i32) {
        self.amount_scrolled += amount as f32;
        self.clamp_scroll();
        self.initial_click_y = -2;
    }

    /// One mouse-wheel notch. Vanilla normalises any wheel delta to ±1 and scrolls
    /// `slot_height / 2` pixels — so the step is the same however coarse the wheel is.
    // java: GuiSlot.java:411
    pub fn mouse_wheel(&mut self, delta: f32) {
        if delta == 0.0 {
            return;
        }
        let dir = if delta > 0.0 { -1 } else { 1 };
        self.amount_scrolled += (dir * self.slot_height / 2) as f32;
        // GuiSlot.handleMouseInput leaves this temporarily out of range. `drawScreen` clamps at
        // the beginning of the next frame, after all queued input has been consumed.
    }

    /// Whether a pointer is inside the list viewport, including all four boundary pixels.
    // java: GuiSlot.java:188 (isMouseYWithinSlotBounds; despite its name it also checks x)
    #[must_use]
    pub fn is_mouse_within_bounds(&self, x: i32, y: i32) -> bool {
        (self.left..=self.right).contains(&x) && (self.top..=self.bottom).contains(&y)
    }

    /// Apply a wheel event only inside the viewport, as `GuiSlot.handleMouseInput` does.
    pub fn mouse_wheel_at(&mut self, x: i32, y: i32, delta: f32) {
        if self.is_mouse_within_bounds(x, y) {
            self.mouse_wheel(delta);
        }
    }

    /// Move the viewport sideways: `left = origin`, `right = origin + width`.
    ///
    /// The only way vanilla positions a list anywhere but flush against the screen's left edge,
    /// and the missing half of laying out `GuiScreenResourcePacks`' two 200-wide off-centre
    /// columns (`setSlotXBoundsFromLeft(width / 2 - 4 - 200)` and `(width / 2 + 4)`). Note
    /// [`set_dimensions`](Self::set_dimensions) resets both edges, exactly as vanilla's does, so
    /// call this *after* it.
    // java: GuiSlot.java:515 (setSlotXBoundsFromLeft)
    pub fn set_slot_x_bounds_from_left(&mut self, origin: i32) {
        self.left = origin;
        self.right = origin + self.width;
    }

    /// Left edge of the scrollbar track, per [`scrollbar_placement`](Self::scrollbar_placement).
    // java: GuiSlot.java:489 (getScrollBarX) and its three overrides
    #[must_use]
    pub fn scrollbar_x(&self) -> i32 {
        match self.scrollbar_placement {
            ScrollbarX::Centered { offset } => self.width / 2 + DEFAULT_SCROLLBAR_OFFSET + offset,
            ScrollbarX::FromRight { inset } => self.right - inset,
            ScrollbarX::Fixed(x) => x,
        }
    }

    /// Left and right edges of the content column.
    ///
    /// Vanilla centres the column on the viewport — `left + width / 2 ± listWidth / 2` — so the
    /// *origin* is [`left`](Self::left) (see
    /// [`set_slot_x_bounds_from_left`](Self::set_slot_x_bounds_from_left)) and the *width* is
    /// [`list_width`](Self::list_width). A resource-pack column, whose `getListWidth()` returns
    /// the viewport width, therefore spans exactly `left..left + width`.
    // java: GuiSlot.java:151
    #[must_use]
    pub fn content_bounds(&self) -> (i32, i32) {
        (
            self.left + self.width / 2 - self.list_width / 2,
            self.left + self.width / 2 + self.list_width / 2,
        )
    }

    /// The entry under `(x, y)`, if any.
    ///
    /// Note the bounds are **inclusive** on both sides here, unlike a button's — that is
    /// vanilla, and it is why the pixel column at the right edge of a list entry is clickable
    /// while the equivalent column of a button is not.
    // java: GuiSlot.java:149 (getSlotIndexFromScreenCoords)
    #[must_use]
    pub fn slot_index_at(&self, x: i32, y: i32) -> Option<usize> {
        if self.slot_height <= 0 {
            return None;
        }
        let (i, j) = self.content_bounds();
        let k = y - self.top - self.header_padding + self.scroll_pixels() - 4;
        let l = k / self.slot_height;
        if x < self.scrollbar_x()
            && x >= i
            && x <= j
            && l >= 0
            && k >= 0
            && (l as usize) < self.size
        {
            Some(l as usize)
        } else {
            None
        }
    }

    /// Where entry `index` draws, in screen coordinates.
    ///
    /// The rectangle is `slot_height - 4` tall (vanilla's inter-entry gap) and may be entirely
    /// off-screen; compare against [`ScrollList::top`]/[`ScrollList::bottom`] before drawing.
    // java: GuiSlot.java:450 (drawSelectionBox)
    #[must_use]
    pub fn slot_rect(&self, index: usize) -> Rect {
        let x = self.left + self.width / 2 - self.list_width / 2 + 2;
        let y = self.top + 4 - self.scroll_pixels()
            + index as i32 * self.slot_height
            + self.header_padding;
        // `drawSelectionBox` passes `getListWidth()` unchanged as drawEntry's width. Only the
        // row origin receives the `+ 2`; reducing the width by four made entry-local right-edge
        // artwork four pixels too far left.
        Rect::new(x, y, self.list_width, self.slot_height - 4)
    }

    /// Whether entry `index` overlaps the viewport at all.
    #[must_use]
    pub fn slot_visible(&self, index: usize) -> bool {
        let r = self.slot_rect(index);
        r.y <= self.bottom && r.bottom() >= self.top
    }

    /// The indices that overlap the viewport, in draw order.
    #[must_use]
    pub fn visible_slots(&self) -> Vec<usize> {
        (0..self.size).filter(|i| self.slot_visible(*i)).collect()
    }

    /// The scrollbar's geometry, or `None` when the content fits and no bar is drawn.
    // java: GuiSlot.java:275
    #[must_use]
    pub fn scrollbar(&self) -> Option<Scrollbar> {
        let max = self.max_scroll();
        let content = self.content_height();
        if max <= 0 || content <= 0 {
            return None;
        }
        let span = self.bottom - self.top;
        let mut thumb = span * span / content;
        thumb = clamp_i32(thumb, MIN_THUMB_HEIGHT, span - 8);
        let mut thumb_top = self.scroll_pixels() * (span - thumb) / max + self.top;
        if thumb_top < self.top {
            thumb_top = self.top;
        }
        Some(Scrollbar {
            x: self.scrollbar_x(),
            track_top: self.top,
            track_bottom: self.bottom,
            thumb_top,
            thumb_height: thumb,
        })
    }

    /// Begin a drag. `y` is the mouse position; returns whether the press started a scroll drag
    /// on the bar rather than on the content.
    ///
    /// Vanilla computes a *negative reciprocal* multiplier when the press lands on the
    /// scrollbar, so dragging the thumb one pixel scrolls the content by the ratio of content
    /// to track — which is why bar drags feel proportional and content drags feel 1:1.
    // java: GuiSlot.java:338
    pub fn mouse_pressed(&mut self, x: i32, y: i32) -> bool {
        if !self.is_mouse_within_bounds(x, y) {
            self.initial_click_y = -2;
            return false;
        }
        // GuiSlot's input loop marks the press as handled (and leaves
        // `initialClickY = -2`) when it lands in the non-entry area above the
        // first row.  This is the four-pixel top fade for an ordinary list and
        // also includes a list header.  A click outside the content column can
        // still begin a one-to-one drag, just as vanilla's `flag1` branch does.
        let (content_left, content_right) = self.content_bounds();
        let content_y = y - self.top - self.header_padding + self.scroll_pixels() - 4;
        if x >= content_left && x <= content_right && content_y < 0 {
            self.initial_click_y = -2;
            return false;
        }
        let on_bar = x >= self.scrollbar_x() && x <= self.scrollbar_x() + SCROLLBAR_WIDTH;
        if on_bar {
            self.scroll_multiplier = -1.0;
            let max = self.max_scroll().max(1);
            let span = self.bottom - self.top;
            let content = self.content_height().max(1);
            let mut thumb = (span * span) as f32 / content as f32;
            thumb = clamp_i32(thumb as i32, MIN_THUMB_HEIGHT, span - 8) as f32;
            let denom = (span as f32 - thumb) / max as f32;
            if denom != 0.0 {
                self.scroll_multiplier /= denom;
            }
        } else {
            self.scroll_multiplier = 1.0;
        }
        self.initial_click_y = y;
        on_bar
    }

    /// Continue a drag started by [`mouse_pressed`](Self::mouse_pressed).
    // java: GuiSlot.java:400
    pub fn mouse_dragged(&mut self, y: i32) {
        if !self.enabled || self.initial_click_y < 0 {
            return;
        }
        self.amount_scrolled -= (y - self.initial_click_y) as f32 * self.scroll_multiplier;
        self.initial_click_y = y;
        // java: GuiSlot.handleMouseInput leaves this value out of range. The
        // clamp runs at the beginning of the next drawScreen, after all input
        // events in this pump have been consumed. Keeping the transient value
        // matters when a gesture reverses direction before that draw.
    }

    /// End a drag.
    pub fn mouse_released(&mut self) {
        self.initial_click_y = -1;
    }

    /// Draw the tiled background, the top/bottom fades and the scrollbar, in that order.
    ///
    /// **This is not what a screen wants.** Vanilla's `drawScreen` puts the entries *between*
    /// the two halves — background, entries, then the occluding bands and the bar — so a screen
    /// must call [`draw_background`](Self::draw_background), draw its entries, and then call
    /// [`draw_overlay`](Self::draw_overlay). Calling this instead paints the opaque background
    /// over entries already drawn, which is invisible in a vertex-count test and total on
    /// screen. It is kept because the two halves back to back are exactly the byte sequence the
    /// geometry fingerprint below was measured against.
    // java: GuiSlot.java:222 (drawScreen)
    pub fn draw_frame(&mut self, d: &mut DrawList) {
        self.draw_background(d);
        self.draw_overlay(d);
    }

    /// The first half of vanilla's `drawScreen`: the scrolling tiled background.
    ///
    /// Call this, then draw entries into each [`slot_rect`](Self::slot_rect) returned by
    /// [`visible_slots`](Self::visible_slots), then call [`draw_overlay`](Self::draw_overlay).
    // java: GuiSlot.java:222 (drawScreen) — the `drawContainerBackground` half
    pub fn draw_background(&mut self, d: &mut DrawList) {
        self.clamp_scroll();
        let dirt = UiTexture::gui(OPTIONS_BACKGROUND);
        // The background scrolls with the content, tinted 0x202020.
        d.tiled(
            &dirt,
            Rect::new(
                self.left,
                self.top,
                self.right - self.left,
                self.bottom - self.top,
            ),
            32.0,
            self.scroll_pixels() as f32 / 32.0,
            0xFF20_2020,
        );
    }

    /// The second half of vanilla's `drawScreen`: the bands that clip entries at the viewport
    /// seams, the two gradient fades, and the scrollbar. Draw entries before this.
    // java: GuiSlot.java:222 (drawScreen) — everything after the slot loop
    pub fn draw_overlay(&mut self, d: &mut DrawList) {
        let dirt = UiTexture::gui(OPTIONS_BACKGROUND);
        // Bands above and below the viewport hide entries that scrolled out.
        d.tiled(
            &dirt,
            Rect::new(self.left, 0, self.width, self.top),
            32.0,
            0.0,
            0xFF40_4040,
        );
        d.tiled(
            &dirt,
            Rect::new(
                self.left,
                self.bottom,
                self.width,
                self.height - self.bottom,
            ),
            32.0,
            0.0,
            0xFF40_4040,
        );
        // Four-pixel gradient fades at both seams.
        d.gradient_rect(
            self.left,
            self.top,
            self.right,
            self.top + 4,
            0xFF00_0000,
            0x0000_0000,
        );
        d.gradient_rect(
            self.left,
            self.bottom - 4,
            self.right,
            self.bottom,
            0x0000_0000,
            0xFF00_0000,
        );

        if let Some(bar) = self.scrollbar() {
            d.rect(
                bar.x,
                bar.track_top,
                bar.x + SCROLLBAR_WIDTH,
                bar.track_bottom,
                0xFF00_0000,
            );
            d.rect(
                bar.x,
                bar.thumb_top,
                bar.x + SCROLLBAR_WIDTH,
                bar.thumb_top + bar.thumb_height,
                0xFF80_8080,
            );
            d.rect(
                bar.x,
                bar.thumb_top,
                bar.x + SCROLLBAR_WIDTH - 1,
                bar.thumb_top + bar.thumb_height - 1,
                0xFFC0_C0C0,
            );
        }
    }

    /// Draw the selection outline around entry `index`.
    // java: GuiSlot.java:466
    pub fn draw_selection_box(&self, d: &mut DrawList, index: usize) {
        if !self.show_selection_box {
            return;
        }
        let r = self.slot_rect(index);
        let (i1, j1) = self.content_bounds();
        d.rect(i1, r.y - 2, j1, r.bottom() + 2, 0xFF80_8080);
        d.rect(i1 + 1, r.y - 1, j1 - 1, r.bottom() + 1, 0xFF00_0000);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::widget::test_support::fnv1a;

    /// A 400-wide viewport from y=32 to y=232, entries 36 tall — the shape of vanilla's server
    /// list at guiScale 2.
    fn list(entries: usize) -> ScrollList {
        let mut l = ScrollList::new(400, 264, 32, 232, 36);
        l.size = entries;
        l
    }

    /// Every observable of a default-geometry list, over a matrix of shapes and scroll states.
    ///
    /// Only pre-existing API, so this exact function was run against the tree as it stood before
    /// [`ScrollbarX`] existed, to capture the golden below.
    fn default_geometry_fingerprint() -> u64 {
        let mut out: Vec<u8> = Vec::new();
        fn put(out: &mut Vec<u8>, n: i64) {
            out.extend_from_slice(&n.to_le_bytes());
        }

        for &(w, h, top, bottom, sh) in &[
            (400i32, 264i32, 32i32, 232i32, 36i32),
            (427, 240, 32, 208, 18),
            (640, 360, 63, 305, 20),
            (320, 240, 32, 232, 25),
            (401, 265, 33, 231, 36),
            (200, 400, 32, 349, 36),
        ] {
            for &size in &[0usize, 1, 3, 6, 10, 100, 10_000] {
                for &(header, pad) in &[(false, 0i32), (true, 20), (true, 13)] {
                    for &lw in &[DEFAULT_LIST_WIDTH, 305, 252, 200] {
                        let mut l = ScrollList::new(w, h, top, bottom, sh);
                        l.size = size;
                        l.set_header(header, pad);
                        l.list_width = lw;
                        for &scroll in &[0i32, 17, 4000] {
                            l.scroll_by(scroll);
                            put(&mut out, i64::from(l.scrollbar_x()));
                            let (a, b) = l.content_bounds();
                            put(&mut out, i64::from(a));
                            put(&mut out, i64::from(b));
                            put(&mut out, i64::from(l.max_scroll()));
                            put(&mut out, i64::from(l.content_height()));
                            put(&mut out, i64::from(l.scroll_pixels()));
                            match l.scrollbar() {
                                Some(s) => {
                                    put(&mut out, 1);
                                    put(&mut out, i64::from(s.x));
                                    put(&mut out, i64::from(s.track_top));
                                    put(&mut out, i64::from(s.track_bottom));
                                    put(&mut out, i64::from(s.thumb_top));
                                    put(&mut out, i64::from(s.thumb_height));
                                }
                                None => put(&mut out, 0),
                            }
                            for i in [0usize, 1, 2, size.saturating_sub(1)] {
                                let r = l.slot_rect(i);
                                put(&mut out, i64::from(r.x));
                                put(&mut out, i64::from(r.y));
                                put(&mut out, i64::from(r.width));
                                put(&mut out, i64::from(r.height));
                                put(&mut out, i64::from(l.slot_visible(i)));
                            }
                            put(&mut out, l.visible_slots().len() as i64);
                            // The hit test, swept across the whole viewport including the
                            // scrollbar column and both content edges.
                            for x in (0..=w).step_by(7) {
                                for y in (top - 8..=bottom + 8).step_by(11) {
                                    put(
                                        &mut out,
                                        match l.slot_index_at(x, y) {
                                            Some(i) => i as i64,
                                            None => -1,
                                        },
                                    );
                                }
                            }
                            let mut d = DrawList::new();
                            l.draw_frame(&mut d);
                            l.draw_selection_box(&mut d, 0);
                            out.extend_from_slice(&d.vertex_bytes());
                            put(&mut out, d.batches().len() as i64);
                        }
                    }
                }
            }
        }
        fnv1a(&out)
    }

    /// Whole-widget regression fingerprint after restoring vanilla's full `getListWidth()`
    /// draw-entry width. This deliberately differs from the pre-fix fingerprint, whose row
    /// rectangles were four pixels short; the focused width test below states that amendment.
    #[test]
    fn default_geometry_matches_the_full_entry_width_tree() {
        assert_eq!(default_geometry_fingerprint(), 11_441_919_784_047_922_484);
    }

    #[test]
    fn max_scroll_accounts_for_vanillas_four_pixel_fudge() {
        let mut l = list(0);
        assert_eq!(l.max_scroll(), 0);
        // Viewport is 200 tall, so 196 pixels of content fit "for free".
        l.size = 5; // 180
        assert_eq!(l.content_height(), 180);
        assert_eq!(l.max_scroll(), 0);
        l.size = 6; // 216
        assert_eq!(l.max_scroll(), 216 - 196);
        l.size = 100;
        assert_eq!(l.max_scroll(), 3600 - 196);
    }

    #[test]
    fn scrolling_is_clamped_at_both_ends() {
        let mut l = list(100);
        l.scroll_by(-500);
        assert_eq!(l.amount_scrolled(), 0.0);
        l.scroll_by(100_000);
        assert_eq!(l.amount_scrolled(), l.max_scroll() as f32);
    }

    #[test]
    fn the_wheel_moves_half_a_slot_per_notch_whatever_the_delta() {
        let mut l = list(100);
        l.mouse_wheel(-1.0);
        assert_eq!(l.amount_scrolled(), 18.0, "slot_height / 2");
        l.mouse_wheel(-120.0);
        assert_eq!(l.amount_scrolled(), 36.0, "a coarse wheel scrolls the same");
        l.mouse_wheel(1.0);
        assert_eq!(l.amount_scrolled(), 18.0);
        l.mouse_wheel(0.0);
        assert_eq!(l.amount_scrolled(), 18.0);
    }

    #[test]
    fn wheel_overscroll_is_deferred_until_the_next_draw() {
        let mut l = list(100);
        let max = l.max_scroll();
        l.scroll_by(max);
        l.mouse_wheel(-1.0);
        assert_eq!(
            l.amount_scrolled(),
            (max + l.slot_height / 2) as f32,
            "an eager clamp mutant loses vanilla's between-event state"
        );

        let mut d = DrawList::new();
        l.draw_background(&mut d);
        assert_eq!(l.amount_scrolled(), max as f32);
    }

    #[test]
    fn wheel_and_drag_input_are_bounded_on_both_axes() {
        let mut l = list(100);
        l.mouse_wheel_at(l.left - 1, l.top, -1.0);
        l.mouse_wheel_at(l.right + 1, l.top, -1.0);
        l.mouse_wheel_at(l.left, l.top - 1, -1.0);
        l.mouse_wheel_at(l.left, l.bottom + 1, -1.0);
        assert_eq!(l.amount_scrolled(), 0.0);
        assert!(!l.mouse_pressed(l.right + 1, l.top));

        l.mouse_wheel_at(l.right, l.bottom, -1.0);
        assert_eq!(l.amount_scrolled(), 18.0, "all four edges are inclusive");
    }

    #[test]
    fn slot_index_maps_screen_coordinates_back_to_entries() {
        let l = list(10);
        let (left, right) = l.content_bounds();
        assert_eq!((left, right), (90, 310));
        // First entry starts 4 px below the top.
        assert_eq!(l.slot_index_at(200, 32 + 4), Some(0));
        assert_eq!(l.slot_index_at(200, 32 + 4 + 35), Some(0));
        assert_eq!(l.slot_index_at(200, 32 + 4 + 36), Some(1));
        assert_eq!(l.slot_index_at(200, 32), None, "above the first entry");
        // Horizontal bounds are inclusive on both sides.
        assert_eq!(l.slot_index_at(left, 40), Some(0));
        assert_eq!(l.slot_index_at(right, 40), Some(0));
        assert_eq!(l.slot_index_at(left - 1, 40), None);
        assert_eq!(l.slot_index_at(right + 1, 40), None);
        // Past the last entry.
        assert_eq!(l.slot_index_at(200, 32 + 4 + 36 * 10), None);
    }

    #[test]
    fn the_scrollbar_column_is_never_a_slot_click() {
        let l = list(10);
        let bar = l.scrollbar_x();
        assert_eq!(bar, 324);
        assert_eq!(l.slot_index_at(bar, 40), None);
        assert_eq!(
            l.slot_index_at(bar - 1, 40),
            None,
            "still outside the content column"
        );
    }

    #[test]
    fn slot_indices_shift_with_the_scroll_offset() {
        let mut l = list(100);
        assert_eq!(l.slot_index_at(200, 36), Some(0));
        l.scroll_by(36);
        assert_eq!(l.slot_index_at(200, 36), Some(1), "one entry scrolled past");
        assert_eq!(l.slot_rect(1).y, 32 + 4);
    }

    #[test]
    fn entry_width_is_the_full_get_list_width_value() {
        let mut l = list(10);
        l.list_width = 305;
        let row = l.slot_rect(0);
        assert_eq!(row.width, 305);
        assert_eq!(row.x, l.left + l.width / 2 - 305 / 2 + 2);
        assert_eq!(
            row.right(),
            row.x + 305,
            "a list_width - 4 mutant ends four pixels early"
        );
    }

    #[test]
    fn no_scrollbar_when_everything_fits() {
        let mut l = list(3);
        assert!(l.scrollbar().is_none());
        l.size = 100;
        assert!(l.scrollbar().is_some());
    }

    #[test]
    fn the_thumb_shrinks_with_the_content_but_never_below_thirty_two() {
        let mut l = list(20);
        let span = l.bottom - l.top; // 200
        let bar = l.scrollbar().expect("bar");
        // 200*200 / 720 = 55
        assert_eq!(bar.thumb_height, span * span / l.content_height());
        assert_eq!(bar.x, 324);
        assert_eq!(bar.track_top, 32);
        assert_eq!(bar.track_bottom, 232);

        l.size = 10_000;
        let bar = l.scrollbar().expect("bar");
        assert_eq!(bar.thumb_height, MIN_THUMB_HEIGHT, "clamped at the floor");

        // And never taller than the track minus 8.
        l.size = 6;
        let bar = l.scrollbar().expect("bar");
        assert!(bar.thumb_height <= span - 8);
    }

    #[test]
    fn the_thumb_travels_the_whole_track() {
        let mut l = list(100);
        let top = l.scrollbar().expect("bar").thumb_top;
        assert_eq!(top, l.top, "at rest the thumb is flush with the top");
        l.scroll_by(l.max_scroll());
        let bar = l.scrollbar().expect("bar");
        assert_eq!(
            bar.thumb_top + bar.thumb_height,
            l.bottom,
            "fully scrolled, the thumb is flush with the bottom"
        );
    }

    #[test]
    fn dragging_the_content_is_one_to_one_and_dragging_the_bar_is_proportional() {
        let mut l = list(100);
        assert!(!l.mouse_pressed(200, 100), "content drag");
        l.mouse_dragged(80);
        assert_eq!(l.amount_scrolled(), 20.0, "dragging up scrolls down 1:1");

        let mut l = list(100);
        assert!(l.mouse_pressed(l.scrollbar_x() + 2, 100), "bar drag");
        l.mouse_dragged(120);
        // 20 pixels of thumb travel over a (200 - 32) track covering 3404 pixels of content.
        let expected = 20.0 * (3404.0 / (200.0 - 32.0));
        assert!(
            (l.amount_scrolled() - expected).abs() < 1.0,
            "{} vs {expected}",
            l.amount_scrolled()
        );
    }

    #[test]
    fn drag_overscroll_is_deferred_until_the_next_draw() {
        let mut l = list(100);
        assert!(!l.mouse_pressed(200, 100), "content drag");
        l.mouse_dragged(10_000);
        assert!(
            l.amount_scrolled() < 0.0,
            "GuiSlot keeps the transient out-of-range drag value"
        );

        // drawScreen clamps only after the complete input pump has run.
        let mut d = DrawList::new();
        l.draw_background(&mut d);
        assert_eq!(l.amount_scrolled(), 0.0);

        let max = l.max_scroll();
        l.scroll_by(max);
        assert!(!l.mouse_pressed(200, 100), "content drag at the lower end");
        l.mouse_dragged(-10_000);
        assert!(
            l.amount_scrolled() > max as f32,
            "the lower-end drag also remains transiently out of range"
        );
        l.draw_background(&mut d);
        assert_eq!(l.amount_scrolled(), max as f32);
    }

    #[test]
    fn the_top_gap_and_header_do_not_arm_a_content_drag() {
        let mut ordinary = list(100);
        assert!(!ordinary.mouse_pressed(200, ordinary.top));
        ordinary.mouse_dragged(ordinary.top - 40);
        assert_eq!(ordinary.amount_scrolled(), 0.0);

        let mut headed = list(100);
        headed.set_header(true, 20);
        let header_y = headed.top + 4 + 10;
        assert!(!headed.mouse_pressed(200, header_y));
        headed.mouse_dragged(header_y - 40);
        assert_eq!(headed.amount_scrolled(), 0.0);

        // The side gutter remains draggable: vanilla only suppresses the
        // header branch while x is inside the list content bounds.
        assert!(!headed.mouse_pressed(headed.left, header_y));
        headed.mouse_dragged(header_y - 20);
        assert_eq!(headed.amount_scrolled(), 20.0);
    }

    #[test]
    fn a_press_outside_the_viewport_does_not_arm_a_drag() {
        let mut l = list(100);
        assert!(!l.mouse_pressed(200, 5));
        l.mouse_dragged(200);
        assert_eq!(l.amount_scrolled(), 0.0);
    }

    #[test]
    fn a_disabled_list_ignores_drags_but_still_takes_the_wheel() {
        let mut l = list(100);
        l.enabled = false;
        l.mouse_pressed(200, 100);
        l.mouse_dragged(50);
        assert_eq!(l.amount_scrolled(), 0.0);
        l.mouse_wheel(-1.0);
        assert_eq!(l.amount_scrolled(), 18.0);
    }

    #[test]
    fn visible_slots_are_the_ones_overlapping_the_viewport() {
        let mut l = list(100);
        let v = l.visible_slots();
        assert_eq!(v.first(), Some(&0));
        // 200 px of viewport / 36 px per entry -> six entries, plus partials at each end.
        assert!(v.len() >= 6 && v.len() <= 8, "{} visible", v.len());
        l.scroll_by(l.max_scroll());
        let v = l.visible_slots();
        assert_eq!(v.last(), Some(&99));
    }

    /// The bug the amendment exists to fix, stated as vanilla states it.
    ///
    /// `ServerSelectionList` widens the content to 305 *and* moves the bar to `+30`. Doing only
    /// the first — which is all the widget could express before — leaves the 28 pixel columns
    /// between the default bar and the row's right edge dead, on every row of the server list.
    // java: ServerSelectionList.java:91 + :99
    #[test]
    fn the_server_lists_rightmost_columns_are_dead_until_the_scrollbar_moves_with_them() {
        let mut l = list(10);
        l.list_width = DEFAULT_LIST_WIDTH + 85; // ServerSelectionList.getListWidth()
        let (_, right) = l.content_bounds();
        let y = l.slot_rect(0).y + 5;

        // With GuiSlot's own bar the row is 305 wide but only clickable to `width / 2 + 123`.
        assert_eq!(l.scrollbar_x(), 324);
        assert_eq!(right, 352);
        assert_eq!(l.slot_index_at(right, y), None);
        let dead = (l.scrollbar_x()..=right)
            .filter(|x| l.slot_index_at(*x, y).is_none())
            .count();
        assert_eq!(
            dead, 29,
            "324..=352 inclusive, the 28 the report names plus 352"
        );

        // ServerSelectionList's override, and every column of the row answers.
        l.scrollbar_placement = ScrollbarX::Centered { offset: 30 };
        assert_eq!(l.scrollbar_x(), 354);
        for x in l.content_bounds().0..=right {
            assert_eq!(
                l.slot_index_at(x, y),
                Some(0),
                "column {x} should hit row 0"
            );
        }
        assert_eq!(l.slot_index_at(right + 1, y), None, "and it stops there");
    }

    /// `GuiKeyBindingList`: `+15` and a 252-wide content column.
    // java: GuiKeyBindingList.java:63 + :71
    #[test]
    fn the_key_binding_list_uses_its_own_offset_and_width() {
        let mut l = ScrollList::new(400, 264, 63, 232, 20);
        l.size = 40;
        l.list_width = DEFAULT_LIST_WIDTH + 32;
        l.scrollbar_placement = ScrollbarX::Centered { offset: 15 };
        assert_eq!(l.scrollbar_x(), 400 / 2 + 124 + 15);
        assert_eq!(l.content_bounds(), (74, 326));
        let y = l.slot_rect(0).y + 2;
        assert_eq!(
            l.slot_index_at(326, y),
            Some(0),
            "the right edge is inclusive"
        );
        assert_eq!(l.scrollbar().expect("a bar").x, 339);
    }

    /// `GuiScreenResourcePacks`: two 200-wide lists placed off centre, each with its bar six
    /// pixels in from its **own** right edge rather than anywhere near the screen's middle.
    // java: GuiScreenResourcePacks.java:69-73 + GuiResourcePackList.java:55/60
    #[test]
    fn the_two_resource_pack_columns_sit_off_centre_with_their_own_scrollbars() {
        let screen_width = 427;
        let mut columns = Vec::new();
        for origin in [screen_width / 2 - 4 - 200, screen_width / 2 + 4] {
            let mut l = ScrollList::new(200, 240, 32, 189, 36);
            l.size = 8;
            l.list_width = l.width; // GuiResourcePackList.getListWidth() == this.width
            l.scrollbar_placement = ScrollbarX::FromRight { inset: 6 };
            l.set_slot_x_bounds_from_left(origin);
            columns.push(l);
        }

        let (left_col, right_col) = (&columns[0], &columns[1]);
        assert_eq!((left_col.left, left_col.right), (9, 209));
        assert_eq!((right_col.left, right_col.right), (217, 417));
        // The content column really is the whole viewport, off centre.
        assert_eq!(left_col.content_bounds(), (9, 209));
        assert_eq!(right_col.content_bounds(), (217, 417));
        // Each bar tracks its own right edge, and neither is at `width / 2 + 124` = 224.
        assert_eq!(left_col.scrollbar_x(), 203);
        assert_eq!(right_col.scrollbar_x(), 411);

        // A click in the left column hits the left column and misses the right one.
        let y = left_col.slot_rect(0).y + 4;
        assert_eq!(left_col.slot_index_at(100, y), Some(0));
        assert_eq!(right_col.slot_index_at(100, y), None);
        assert_eq!(right_col.slot_index_at(300, y), Some(0));
        assert_eq!(left_col.slot_index_at(300, y), None);
        // The gap between the two columns belongs to neither.
        for x in 210..217 {
            assert_eq!(left_col.slot_index_at(x, y), None, "gap column {x}");
            assert_eq!(right_col.slot_index_at(x, y), None, "gap column {x}");
        }
    }

    #[test]
    fn set_dimensions_resets_the_origin_exactly_as_vanilla_does() {
        let mut l = list(10);
        l.set_slot_x_bounds_from_left(50);
        assert_eq!((l.left, l.right), (50, 450));
        // java: GuiSlot.setDimensions writes `left = 0; right = widthIn` unconditionally.
        l.set_dimensions(400, 264, 32, 232);
        assert_eq!((l.left, l.right), (0, 400));
    }

    #[test]
    fn a_fixed_scrollbar_column_is_taken_literally() {
        let mut l = list(10);
        l.scrollbar_placement = ScrollbarX::Fixed(7);
        assert_eq!(l.scrollbar_x(), 7);
        assert_eq!(l.scrollbar().expect("a bar").x, 7);
        assert!(l.mouse_pressed(9, 100), "the drag still finds the bar");
    }

    #[test]
    fn the_default_placement_is_gui_slots_own() {
        assert_eq!(ScrollbarX::default(), ScrollbarX::Centered { offset: 0 });
        let l = list(10);
        assert_eq!(l.scrollbar_placement, ScrollbarX::default());
        assert_eq!(l.scrollbar_x(), l.width / 2 + DEFAULT_SCROLLBAR_OFFSET);
    }

    #[test]
    fn a_header_pushes_the_first_entry_down() {
        let mut l = list(10);
        l.set_header(true, 20);
        assert_eq!(l.header_padding, 20);
        assert_eq!(l.content_height(), 380);
        assert_eq!(l.slot_rect(0).y, 32 + 4 + 20);
        assert_eq!(l.slot_index_at(200, 32 + 4 + 20), Some(0));
        assert_eq!(l.slot_index_at(200, 32 + 4 + 19), None, "inside the header");
        l.set_header(false, 20);
        assert_eq!(l.header_padding, 0, "clearing the flag zeroes the padding");
    }

    #[test]
    fn a_degenerate_list_does_not_divide_by_zero() {
        let mut l = ScrollList::new(400, 264, 32, 232, 0);
        l.size = 10;
        assert_eq!(l.slot_index_at(200, 100), None);
        assert!(l.scrollbar().is_none());
        let mut d = DrawList::new();
        l.draw_frame(&mut d);
        assert!(!d.is_empty());
    }

    #[test]
    fn draw_frame_emits_the_background_fades_and_bar() {
        let mut l = list(100);
        let mut d = DrawList::new();
        l.draw_frame(&mut d);
        // 3 tiled quads + 2 gradients + 3 bar rects = 8 quads = 48 vertices.
        assert_eq!(d.vertices().len(), 8 * 6);
        let mut empty = DrawList::new();
        let mut short = list(1);
        short.draw_frame(&mut empty);
        assert_eq!(empty.vertices().len(), 5 * 6, "no scrollbar");
    }
}
