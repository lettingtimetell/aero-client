//! Where HUD elements sit: a 9-point anchor plus an offset in scaled GUI pixels, and a scale.
//!
//! # The model
//!
//! An anchor names a point on the screen *and* the matching point on the element: `TopLeft`
//! pins the element's top-left corner to the screen's top-left corner, `BottomRight` pins
//! bottom-right to bottom-right, `Center` pins centre to centre. The offset `(x, y)` then moves
//! the element from there, in scaled GUI pixels (positive is right/down).
//!
//! ```text
//! origin = anchor_fraction * screen + offset - anchor_fraction * element_size
//! ```
//!
//! So a right-anchored element grows to the left when its text gets longer, and an element
//! keeps its distance to its nearest edge when the window or GUI scale changes. That is what
//! the HUD editor relies on: [`placement_from_rect`] is the inverse of [`resolve_rect`], picking
//! the anchor nearest the dragged rect (by thirds of the screen) and the offset that reproduces
//! it exactly.

/// Smallest scale the HUD editor allows.
pub const HUD_MIN_SCALE: f32 = 0.5;
/// Largest scale the HUD editor allows.
pub const HUD_MAX_SCALE: f32 = 3.0;

/// One of the nine screen points an element can be pinned to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HudAnchor {
    /// Top-left corner.
    TopLeft,
    /// Middle of the top edge.
    TopCenter,
    /// Top-right corner.
    TopRight,
    /// Middle of the left edge.
    CenterLeft,
    /// Screen centre.
    Center,
    /// Middle of the right edge.
    CenterRight,
    /// Bottom-left corner.
    BottomLeft,
    /// Middle of the bottom edge.
    BottomCenter,
    /// Bottom-right corner.
    BottomRight,
}

impl HudAnchor {
    /// All nine, row by row from the top-left.
    pub const ALL: [HudAnchor; 9] = [
        HudAnchor::TopLeft,
        HudAnchor::TopCenter,
        HudAnchor::TopRight,
        HudAnchor::CenterLeft,
        HudAnchor::Center,
        HudAnchor::CenterRight,
        HudAnchor::BottomLeft,
        HudAnchor::BottomCenter,
        HudAnchor::BottomRight,
    ];

    /// `(fx, fy)`, each `0.0`, `0.5` or `1.0`: the anchor as a fraction of the screen (and of
    /// the element).
    #[must_use]
    pub const fn fractions(self) -> (f32, f32) {
        match self {
            HudAnchor::TopLeft => (0.0, 0.0),
            HudAnchor::TopCenter => (0.5, 0.0),
            HudAnchor::TopRight => (1.0, 0.0),
            HudAnchor::CenterLeft => (0.0, 0.5),
            HudAnchor::Center => (0.5, 0.5),
            HudAnchor::CenterRight => (1.0, 0.5),
            HudAnchor::BottomLeft => (0.0, 1.0),
            HudAnchor::BottomCenter => (0.5, 1.0),
            HudAnchor::BottomRight => (1.0, 1.0),
        }
    }

    /// Stable JSON id, e.g. `"top_left"`.
    #[must_use]
    pub const fn id(self) -> &'static str {
        match self {
            HudAnchor::TopLeft => "top_left",
            HudAnchor::TopCenter => "top_center",
            HudAnchor::TopRight => "top_right",
            HudAnchor::CenterLeft => "center_left",
            HudAnchor::Center => "center",
            HudAnchor::CenterRight => "center_right",
            HudAnchor::BottomLeft => "bottom_left",
            HudAnchor::BottomCenter => "bottom_center",
            HudAnchor::BottomRight => "bottom_right",
        }
    }

    /// Parse [`HudAnchor::id`]. Case-insensitive; `-` and space are accepted for `_`.
    #[must_use]
    pub fn from_id(id: &str) -> Option<HudAnchor> {
        let wanted: String = id
            .trim()
            .chars()
            .map(|c| match c {
                '-' | ' ' => '_',
                c => c.to_ascii_lowercase(),
            })
            .collect();
        HudAnchor::ALL
            .into_iter()
            .find(|anchor| anchor.id() == wanted)
    }

    /// The anchor whose third of the screen contains the point `(fx, fy)`, given as fractions
    /// of the screen size.
    #[must_use]
    pub fn nearest(fx: f32, fy: f32) -> HudAnchor {
        let column = third(fx);
        let row = third(fy);
        HudAnchor::ALL[row * 3 + column]
    }
}

fn third(f: f32) -> usize {
    if !f.is_finite() || f < 1.0 / 3.0 {
        0
    } else if f < 2.0 / 3.0 {
        1
    } else {
        2
    }
}

/// Where one HUD element sits.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HudPlacement {
    /// The screen point (and matching element point) the element is pinned to.
    pub anchor: HudAnchor,
    /// Horizontal offset from the anchor, scaled GUI pixels, positive is right.
    pub x: f32,
    /// Vertical offset from the anchor, scaled GUI pixels, positive is down.
    pub y: f32,
    /// Size multiplier, [`HUD_MIN_SCALE`]..=[`HUD_MAX_SCALE`].
    pub scale: f32,
}

impl HudPlacement {
    /// A placement at scale 1.
    #[must_use]
    pub const fn new(anchor: HudAnchor, x: f32, y: f32) -> HudPlacement {
        HudPlacement {
            anchor,
            x,
            y,
            scale: 1.0,
        }
    }

    /// The same placement at another scale.
    #[must_use]
    pub const fn scaled(self, scale: f32) -> HudPlacement {
        HudPlacement { scale, ..self }
    }

    /// Replace non-finite offsets with zero and clamp the scale into range.
    #[must_use]
    pub fn sanitized(self) -> HudPlacement {
        let finite = |v: f32| if v.is_finite() { v } else { 0.0 };
        let scale = if self.scale.is_finite() {
            self.scale.clamp(HUD_MIN_SCALE, HUD_MAX_SCALE)
        } else {
            1.0
        };
        HudPlacement {
            anchor: self.anchor,
            x: finite(self.x).clamp(-10_000.0, 10_000.0),
            y: finite(self.y).clamp(-10_000.0, 10_000.0),
            scale,
        }
    }
}

/// An axis-aligned rectangle in scaled GUI pixels (floats: elements may sit between pixels when
/// scaled).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct HudRect {
    /// Left edge.
    pub x: f32,
    /// Top edge.
    pub y: f32,
    /// Width.
    pub w: f32,
    /// Height.
    pub h: f32,
}

impl HudRect {
    /// A rectangle from position and size.
    #[must_use]
    pub const fn new(x: f32, y: f32, w: f32, h: f32) -> HudRect {
        HudRect { x, y, w, h }
    }

    /// Right edge.
    #[must_use]
    pub fn right(&self) -> f32 {
        self.x + self.w
    }

    /// Bottom edge.
    #[must_use]
    pub fn bottom(&self) -> f32 {
        self.y + self.h
    }

    /// Centre point.
    #[must_use]
    pub fn center(&self) -> (f32, f32) {
        (self.x + self.w * 0.5, self.y + self.h * 0.5)
    }

    /// Inclusive top-left, exclusive bottom-right hit test.
    #[must_use]
    pub fn contains(&self, px: f32, py: f32) -> bool {
        px >= self.x && py >= self.y && px < self.right() && py < self.bottom()
    }

    /// The same rectangle moved by `(dx, dy)`.
    #[must_use]
    pub fn translated(&self, dx: f32, dy: f32) -> HudRect {
        HudRect {
            x: self.x + dx,
            y: self.y + dy,
            ..*self
        }
    }
}

/// Resolve a placement to the element's on-screen rectangle.
///
/// * `size` is the element's **unscaled** size in GUI pixels (what its layout reports);
/// * `screen` is the scaled GUI resolution.
///
/// The result is clamped onto the screen (an element bigger than the screen is pinned to the
/// top-left), so a placement saved on a bigger window never leaves an element unreachable.
#[must_use]
pub fn resolve_rect(placement: &HudPlacement, size: (f32, f32), screen: (f32, f32)) -> HudRect {
    let placement = placement.sanitized();
    let (fx, fy) = placement.anchor.fractions();
    let w = (size.0 * placement.scale).max(0.0);
    let h = (size.1 * placement.scale).max(0.0);
    let x = fx * screen.0 + placement.x - fx * w;
    let y = fy * screen.1 + placement.y - fy * h;
    HudRect {
        x: clamp_axis(x, w, screen.0),
        y: clamp_axis(y, h, screen.1),
        w,
        h,
    }
}

fn clamp_axis(origin: f32, extent: f32, screen: f32) -> f32 {
    let max = (screen - extent).max(0.0);
    origin.clamp(0.0, max)
}

/// The inverse of [`resolve_rect`] used by dragging: the anchor nearest the rectangle's centre
/// (by thirds of the screen) plus the offset that puts the rectangle exactly where it is.
///
/// `rect` is the element's on-screen (already scaled) rectangle; `scale` is carried through.
#[must_use]
pub fn placement_from_rect(rect: &HudRect, scale: f32, screen: (f32, f32)) -> HudPlacement {
    let (cx, cy) = rect.center();
    let fx_screen = if screen.0 > 0.0 { cx / screen.0 } else { 0.0 };
    let fy_screen = if screen.1 > 0.0 { cy / screen.1 } else { 0.0 };
    let anchor = HudAnchor::nearest(fx_screen, fy_screen);
    let (fx, fy) = anchor.fractions();
    HudPlacement {
        anchor,
        x: rect.x + fx * rect.w - fx * screen.0,
        y: rect.y + fy * rect.h - fy * screen.1,
        scale,
    }
    .sanitized()
}

/// Which guide lines a snap engaged, for the HUD editor to draw. Coordinates in GUI pixels.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct SnapGuides {
    /// A vertical guide at this x (a screen edge or the centre line).
    pub vertical: Option<f32>,
    /// A horizontal guide at this y.
    pub horizontal: Option<f32>,
}

/// Snap a dragged rectangle to the screen edges and centre lines when it is within `threshold`
/// GUI pixels of one. Each axis snaps to at most one target (the closest).
#[must_use]
pub fn snap_rect(rect: HudRect, screen: (f32, f32), threshold: f32) -> (HudRect, SnapGuides) {
    let mut out = rect;
    let mut guides = SnapGuides::default();
    // Per candidate: (the rect's edge or centre coordinate, the origin that puts it exactly on
    // the guide, the guide's coordinate).
    let pick = |origin: f32, extent: f32, screen: f32| -> Option<(f32, f32)> {
        let candidates = [
            (origin, 0.0, 0.0),
            (origin + extent, screen - extent, screen),
            (origin + extent * 0.5, (screen - extent) * 0.5, screen * 0.5),
        ];
        candidates
            .iter()
            .map(|&(at, snapped, guide)| ((at - guide).abs(), snapped, guide))
            .filter(|(distance, _, _)| *distance <= threshold)
            .min_by(|a, b| a.0.total_cmp(&b.0))
            .map(|(_, snapped, guide)| (snapped, guide))
    };
    if let Some((x, guide)) = pick(rect.x, rect.w, screen.0) {
        out.x = x;
        guides.vertical = Some(guide);
    }
    if let Some((y, guide)) = pick(rect.y, rect.h, screen.1) {
        out.y = y;
        guides.horizontal = Some(guide);
    }
    (out, guides)
}

/// The HUD editor's snapping: [`snap_rect`]'s screen edges and centre lines, plus the 2-pixel
/// margin every default placement uses, plus the left/centre/right and top/middle/bottom lines
/// of `others` (the other visible elements). Per axis the nearest line within `threshold` wins.
#[must_use]
pub fn snap_rect_with(
    rect: HudRect,
    screen: (f32, f32),
    threshold: f32,
    others: &[HudRect],
) -> (HudRect, SnapGuides) {
    let mut out = rect;
    let mut guides = SnapGuides::default();
    if let Some((x, guide)) = snap_axis(
        rect.x,
        rect.w,
        screen.0,
        threshold,
        others.iter().map(|o| (o.x, o.w)),
    ) {
        out.x = x;
        guides.vertical = Some(guide);
    }
    if let Some((y, guide)) = snap_axis(
        rect.y,
        rect.h,
        screen.1,
        threshold,
        others.iter().map(|o| (o.y, o.h)),
    ) {
        out.y = y;
        guides.horizontal = Some(guide);
    }
    (out, guides)
}

/// One axis of [`snap_rect_with`]: the new origin and the guide coordinate, if any line is
/// within `threshold`.
fn snap_axis(
    origin: f32,
    extent: f32,
    screen: f32,
    threshold: f32,
    others: impl Iterator<Item = (f32, f32)>,
) -> Option<(f32, f32)> {
    let mut best: Option<(f32, f32, f32)> = None;
    let mut consider = |part: f32, target: f32| {
        let at = origin + extent * part;
        let distance = (at - target).abs();
        if distance <= threshold && best.is_none_or(|b| distance < b.0) {
            best = Some((distance, target - extent * part, target));
        }
    };
    consider(0.0, 0.0);
    consider(0.0, 2.0);
    consider(1.0, screen);
    consider(1.0, screen - 2.0);
    consider(0.5, screen * 0.5);
    for (start, length) in others {
        for part in [0.0, 0.5, 1.0] {
            for target in [0.0, 0.5, 1.0] {
                consider(part, start + length * target);
            }
        }
    }
    best.map(|(_, origin, guide)| (origin, guide))
}

#[cfg(test)]
mod tests {
    use super::*;

    const SCREEN: (f32, f32) = (640.0, 360.0);

    #[test]
    fn snapping_reaches_other_elements_and_the_two_pixel_margin() {
        let other = HudRect::new(100.0, 50.0, 60.0, 16.0);
        // Left edge 2 px from the other element's left edge: aligns with it.
        let (r, g) = snap_rect_with(
            HudRect::new(102.0, 200.0, 40.0, 10.0),
            SCREEN,
            4.0,
            &[other],
        );
        assert_eq!(r.x, 100.0);
        assert_eq!(g.vertical, Some(100.0));
        // Top edge 3 px below the other's bottom: stacks under it.
        let (r, g) = snap_rect_with(HudRect::new(300.0, 69.0, 40.0, 10.0), SCREEN, 4.0, &[other]);
        assert_eq!(r.y, 66.0);
        assert_eq!(g.horizontal, Some(66.0));
        // The 2 px margin line beats nothing; the screen edge itself still works.
        let (r, _) = snap_rect_with(HudRect::new(3.5, 200.0, 40.0, 10.0), SCREEN, 4.0, &[]);
        assert_eq!(r.x, 2.0);
        let (r, _) = snap_rect_with(HudRect::new(0.5, 200.0, 40.0, 10.0), SCREEN, 4.0, &[]);
        assert_eq!(r.x, 0.0);
        // Far from everything: untouched.
        let free = HudRect::new(400.0, 200.0, 40.0, 10.0);
        assert_eq!(snap_rect_with(free, SCREEN, 4.0, &[other]).0, free);
    }

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-3
    }

    #[test]
    fn top_left_offset_is_the_origin() {
        let p = HudPlacement::new(HudAnchor::TopLeft, 4.0, 6.0);
        let r = resolve_rect(&p, (50.0, 12.0), SCREEN);
        assert_eq!(r, HudRect::new(4.0, 6.0, 50.0, 12.0));
    }

    #[test]
    fn bottom_right_grows_left_and_up() {
        let p = HudPlacement::new(HudAnchor::BottomRight, -2.0, -3.0);
        let short = resolve_rect(&p, (40.0, 10.0), SCREEN);
        let long = resolve_rect(&p, (80.0, 10.0), SCREEN);
        assert!(close(short.right(), 638.0) && close(long.right(), 638.0));
        assert!(close(short.bottom(), 357.0));
        assert!(long.x < short.x, "a longer element extends leftwards");
    }

    #[test]
    fn center_anchor_centres_and_scale_multiplies_size() {
        let p = HudPlacement::new(HudAnchor::Center, 0.0, 0.0).scaled(2.0);
        let r = resolve_rect(&p, (30.0, 10.0), SCREEN);
        assert!(close(r.w, 60.0) && close(r.h, 20.0));
        assert!(close(r.center().0, 320.0) && close(r.center().1, 180.0));
    }

    #[test]
    fn off_screen_placements_are_pulled_back_on() {
        let p = HudPlacement::new(HudAnchor::TopLeft, 900.0, -50.0);
        let r = resolve_rect(&p, (40.0, 10.0), SCREEN);
        assert!(close(r.x, 600.0) && close(r.y, 0.0));
        // Bigger than the screen: pinned at the origin rather than going negative.
        let huge = resolve_rect(&p, (1000.0, 1000.0), SCREEN);
        assert_eq!((huge.x, huge.y), (0.0, 0.0));
    }

    #[test]
    fn placement_from_rect_picks_the_nearest_third_and_round_trips() {
        let cases = [
            (HudRect::new(5.0, 5.0, 40.0, 10.0), HudAnchor::TopLeft),
            (HudRect::new(300.0, 2.0, 40.0, 10.0), HudAnchor::TopCenter),
            (HudRect::new(590.0, 4.0, 40.0, 10.0), HudAnchor::TopRight),
            (HudRect::new(3.0, 170.0, 40.0, 10.0), HudAnchor::CenterLeft),
            (HudRect::new(300.0, 175.0, 40.0, 10.0), HudAnchor::Center),
            (
                HudRect::new(598.0, 160.0, 40.0, 10.0),
                HudAnchor::CenterRight,
            ),
            (HudRect::new(1.0, 340.0, 40.0, 10.0), HudAnchor::BottomLeft),
            (
                HudRect::new(310.0, 330.0, 40.0, 10.0),
                HudAnchor::BottomCenter,
            ),
            (
                HudRect::new(580.0, 345.0, 40.0, 10.0),
                HudAnchor::BottomRight,
            ),
        ];
        for (rect, anchor) in cases {
            for scale in [1.0, 0.5, 2.0] {
                let p = placement_from_rect(&rect, scale, SCREEN);
                assert_eq!(p.anchor, anchor, "{rect:?}");
                let unscaled = (rect.w / scale, rect.h / scale);
                let back = resolve_rect(&p, unscaled, SCREEN);
                assert!(
                    close(back.x, rect.x) && close(back.y, rect.y),
                    "{rect:?} at {scale} came back as {back:?}"
                );
            }
        }
    }

    #[test]
    fn a_dragged_element_keeps_its_edge_distance_after_a_resize() {
        // Dropped 4 px from the right edge on a 640-wide screen...
        let rect = HudRect::new(596.0, 100.0, 40.0, 10.0);
        let p = placement_from_rect(&rect, 1.0, SCREEN);
        // ...stays 4 px from the right edge on a 960-wide one.
        let wide = resolve_rect(&p, (40.0, 10.0), (960.0, 540.0));
        assert!(close(960.0 - wide.right(), 4.0), "{wide:?}");
    }

    #[test]
    fn snapping_engages_edges_and_centre_lines_within_the_threshold() {
        let (r, g) = snap_rect(HudRect::new(3.0, 100.0, 40.0, 10.0), SCREEN, 4.0);
        assert_eq!(r.x, 0.0);
        assert_eq!(g.vertical, Some(0.0));
        assert_eq!(g.horizontal, None);

        let (r, g) = snap_rect(HudRect::new(598.0, 348.0, 40.0, 10.0), SCREEN, 4.0);
        assert!(close(r.right(), 640.0) && close(r.bottom(), 360.0));
        assert_eq!(g.vertical, Some(640.0));
        assert_eq!(g.horizontal, Some(360.0));

        let (r, g) = snap_rect(HudRect::new(302.0, 172.0, 40.0, 10.0), SCREEN, 4.0);
        assert!(close(r.center().0, 320.0) && close(r.center().1, 180.0));
        assert_eq!(g.vertical, Some(320.0));

        let (r, g) = snap_rect(HudRect::new(100.0, 100.0, 40.0, 10.0), SCREEN, 4.0);
        assert_eq!(r, HudRect::new(100.0, 100.0, 40.0, 10.0));
        assert_eq!(g, SnapGuides::default());
    }

    #[test]
    fn anchor_ids_round_trip_and_bad_input_is_rejected() {
        for anchor in HudAnchor::ALL {
            assert_eq!(HudAnchor::from_id(anchor.id()), Some(anchor));
        }
        assert_eq!(HudAnchor::from_id("Top-Right"), Some(HudAnchor::TopRight));
        assert_eq!(HudAnchor::from_id("upper left"), None);
    }

    #[test]
    fn sanitize_clamps_scale_and_drops_nan() {
        let p = HudPlacement {
            anchor: HudAnchor::Center,
            x: f32::NAN,
            y: f32::INFINITY,
            scale: 99.0,
        }
        .sanitized();
        assert_eq!((p.x, p.y, p.scale), (0.0, 0.0, HUD_MAX_SCALE));
    }
}
