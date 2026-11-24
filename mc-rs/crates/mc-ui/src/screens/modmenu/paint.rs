//! Anti-aliased vector shapes for the mod menu, emitted as triangles into a [`DrawList`] in
//! physical pixels: rounded rectangles (solid or linear-gradient fill), strokes, soft shadows,
//! circles, lines and filled outlines ([`PathMesh`]). Every edge gets a one-pixel alpha fringe
//! (an anti-aliasing ramp built into the geometry), so no MSAA is needed.
//!
//! [`Painter`] also clips every triangle to an optional axis-aligned rectangle on the CPU
//! (Sutherland–Hodgman), which is how a scrolled window body is cut off at its edges without a
//! scissor in the shared UI renderer.

use crate::widget::{DrawList, UiTexture};

/// Most outline points one rounded rectangle can produce (4 corners × 13 points).
const MAX_OUTLINE: usize = 52;

/// A straight-alpha RGBA colour, `0..=1`.
pub type Rgba = [f32; 4];

/// `rgba(r, g, b, a)` with 0–255 channels and a 0–1 alpha, like CSS.
#[must_use]
pub const fn css(r: u8, g: u8, b: u8, a: f32) -> Rgba {
    [r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0, a]
}

/// White at alpha `a`.
#[must_use]
pub const fn white(a: f32) -> Rgba {
    [1.0, 1.0, 1.0, a]
}

/// Black at alpha `a`.
#[must_use]
pub const fn black(a: f32) -> Rgba {
    [0.0, 0.0, 0.0, a]
}

/// Unpack `0xAARRGGBB`.
#[must_use]
pub fn argb(argb: u32) -> Rgba {
    crate::widget::unpack_argb(argb)
}

fn lerp(a: Rgba, b: Rgba, t: f32) -> Rgba {
    let t = t.clamp(0.0, 1.0);
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
        a[3] + (b[3] - a[3]) * t,
    ]
}

/// A rectangle in physical pixels.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Rect {
    /// Left.
    pub x: f32,
    /// Top.
    pub y: f32,
    /// Width.
    pub w: f32,
    /// Height.
    pub h: f32,
}

impl Rect {
    /// From position and size.
    #[must_use]
    pub const fn new(x: f32, y: f32, w: f32, h: f32) -> Rect {
        Rect { x, y, w, h }
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

    /// Hit test, inclusive top-left.
    #[must_use]
    pub fn contains(&self, px: f32, py: f32) -> bool {
        px >= self.x && py >= self.y && px < self.right() && py < self.bottom()
    }

    /// Shrunk by `d` on every side (negative grows).
    #[must_use]
    pub fn inset(&self, d: f32) -> Rect {
        Rect::new(self.x + d, self.y + d, self.w - 2.0 * d, self.h - 2.0 * d)
    }

    /// Moved by `(dx, dy)`.
    #[must_use]
    pub fn offset(&self, dx: f32, dy: f32) -> Rect {
        Rect::new(self.x + dx, self.y + dy, self.w, self.h)
    }

    /// The overlap of two rectangles (zero-sized when they do not meet).
    #[must_use]
    pub fn intersect(&self, other: &Rect) -> Rect {
        let x0 = self.x.max(other.x);
        let y0 = self.y.max(other.y);
        let x1 = self.right().min(other.right());
        let y1 = self.bottom().min(other.bottom());
        Rect::new(x0, y0, (x1 - x0).max(0.0), (y1 - y0).max(0.0))
    }

    /// Centre point.
    #[must_use]
    pub fn center(&self) -> (f32, f32) {
        (self.x + self.w * 0.5, self.y + self.h * 0.5)
    }
}

/// How a shape is filled.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Fill {
    /// One colour.
    Solid(Rgba),
    /// A two-stop linear gradient from `from` (colour `c0`) to `to` (colour `c1`), in pixels.
    Linear {
        /// Start point.
        from: (f32, f32),
        /// End point.
        to: (f32, f32),
        /// Colour at `from` and before it.
        c0: Rgba,
        /// Colour at `to` and after it.
        c1: Rgba,
    },
}

impl Fill {
    /// A CSS `linear-gradient(<angle>deg, c0, c1)` over `rect` (0deg points up, 90deg right).
    #[must_use]
    pub fn css_angle(rect: Rect, degrees: f32, c0: Rgba, c1: Rgba) -> Fill {
        let a = degrees.to_radians();
        let (dx, dy) = (a.sin(), -a.cos());
        // CSS: the gradient line's length makes the corners hit exactly 0% and 100%.
        let half = (rect.w * dx.abs() + rect.h * dy.abs()) * 0.5;
        let (cx, cy) = rect.center();
        Fill::Linear {
            from: (cx - dx * half, cy - dy * half),
            to: (cx + dx * half, cy + dy * half),
            c0,
            c1,
        }
    }

    fn at(&self, p: [f32; 2]) -> Rgba {
        match *self {
            Fill::Solid(c) => c,
            Fill::Linear { from, to, c0, c1 } => {
                let (dx, dy) = (to.0 - from.0, to.1 - from.1);
                let len2 = dx * dx + dy * dy;
                let t = if len2 > 0.0 {
                    ((p[0] - from.0) * dx + (p[1] - from.1) * dy) / len2
                } else {
                    0.0
                };
                lerp(c0, c1, t)
            }
        }
    }
}

#[derive(Clone, Copy)]
struct Vert {
    pos: [f32; 2],
    uv: [f32; 2],
    color: Rgba,
}

fn mix_vert(a: &Vert, b: &Vert, t: f32) -> Vert {
    Vert {
        pos: [
            a.pos[0] + (b.pos[0] - a.pos[0]) * t,
            a.pos[1] + (b.pos[1] - a.pos[1]) * t,
        ],
        uv: [
            a.uv[0] + (b.uv[0] - a.uv[0]) * t,
            a.uv[1] + (b.uv[1] - a.uv[1]) * t,
        ],
        color: lerp(a.color, b.color, t),
    }
}

/// Emits shapes into a [`DrawList`], with a global alpha, a global offset and an optional clip.
pub struct Painter<'a> {
    /// The target list (physical-pixel space).
    pub list: &'a mut DrawList,
    clip: Option<Rect>,
    /// Multiplies every alpha (window fade).
    pub alpha: f32,
    /// Added to every position (window slide).
    pub offset: (f32, f32),
}

impl<'a> Painter<'a> {
    /// A painter with alpha 1, no offset, no clip.
    pub fn new(list: &'a mut DrawList) -> Painter<'a> {
        Painter {
            list,
            clip: None,
            alpha: 1.0,
            offset: (0.0, 0.0),
        }
    }

    /// Clip subsequent shapes to `rect` (in un-offset coordinates), or stop clipping.
    pub fn set_clip(&mut self, rect: Option<Rect>) {
        self.clip = rect;
    }

    /// The clip in force.
    #[must_use]
    pub fn clip(&self) -> Option<Rect> {
        self.clip
    }

    fn emit(&mut self, texture: &UiTexture, tri: [Vert; 3]) {
        let mut tri = tri;
        for v in &mut tri {
            v.pos[0] += self.offset.0;
            v.pos[1] += self.offset.1;
            v.color[3] *= self.alpha;
        }
        if tri.iter().all(|v| v.color[3] <= 0.0) {
            return;
        }
        // The UI pipeline culls back faces (`DrawList`'s default `UiCull::Back`, vanilla's GUI
        // state), so every triangle must wind like `DrawList::quad`'s: a negative cross product
        // in y-down screen space. The shape builders wind both ways (a fill's fan follows its
        // outline clockwise); fix it once here, before clipping, which preserves winding.
        let cross = (tri[1].pos[0] - tri[0].pos[0]) * (tri[2].pos[1] - tri[0].pos[1])
            - (tri[2].pos[0] - tri[0].pos[0]) * (tri[1].pos[1] - tri[0].pos[1]);
        if cross > 0.0 {
            tri.swap(1, 2);
        }
        let Some(clip) = self.clip.map(|c| c.offset(self.offset.0, self.offset.1)) else {
            self.list.triangle(
                texture,
                [tri[0].pos, tri[1].pos, tri[2].pos],
                [tri[0].uv, tri[1].uv, tri[2].uv],
                [tri[0].color, tri[1].color, tri[2].color],
            );
            return;
        };
        let inside = |v: &Vert| {
            v.pos[0] >= clip.x
                && v.pos[0] <= clip.right()
                && v.pos[1] >= clip.y
                && v.pos[1] <= clip.bottom()
        };
        if tri.iter().all(inside) {
            self.list.triangle(
                texture,
                [tri[0].pos, tri[1].pos, tri[2].pos],
                [tri[0].uv, tri[1].uv, tri[2].uv],
                [tri[0].color, tri[1].color, tri[2].color],
            );
            return;
        }
        // Sutherland–Hodgman against the four clip planes.
        let mut poly: [Vert; 9] = [tri[0]; 9];
        poly[1] = tri[1];
        poly[2] = tri[2];
        let mut count = 3;
        // (axis, limit, keep_greater)
        let planes = [
            (0, clip.x, true),
            (0, clip.right(), false),
            (1, clip.y, true),
            (1, clip.bottom(), false),
        ];
        for (axis, limit, greater) in planes {
            let keep = |v: &Vert| {
                if greater {
                    v.pos[axis] >= limit
                } else {
                    v.pos[axis] <= limit
                }
            };
            let mut out: [Vert; 9] = [poly[0]; 9];
            let mut n = 0;
            for i in 0..count {
                let a = poly[i];
                let b = poly[(i + 1) % count];
                let (ka, kb) = (keep(&a), keep(&b));
                if ka {
                    out[n] = a;
                    n += 1;
                }
                if ka != kb {
                    let t = (limit - a.pos[axis]) / (b.pos[axis] - a.pos[axis]);
                    out[n] = mix_vert(&a, &b, t);
                    n += 1;
                }
                if n >= 9 {
                    break;
                }
            }
            poly = out;
            count = n;
            if count < 3 {
                return;
            }
        }
        for i in 1..count - 1 {
            let (a, b, c) = (poly[0], poly[i], poly[i + 1]);
            self.list.triangle(
                texture,
                [a.pos, b.pos, c.pos],
                [a.uv, b.uv, c.uv],
                [a.color, b.color, c.color],
            );
        }
    }

    fn solid_tri(&mut self, p: [[f32; 2]; 3], c: [Rgba; 3]) {
        let v = |i: usize| Vert {
            pos: p[i],
            uv: [0.0, 0.0],
            color: c[i],
        };
        self.emit(&UiTexture::Solid, [v(0), v(1), v(2)]);
    }

    /// A textured quad with UVs `(u0, v0, u1, v1)`.
    pub fn textured(&mut self, texture: &UiTexture, rect: Rect, uv: [f32; 4], color: Rgba) {
        let (x0, y0, x1, y1) = (rect.x, rect.y, rect.right(), rect.bottom());
        let v = |x: f32, y: f32, u: f32, w: f32| Vert {
            pos: [x, y],
            uv: [u, w],
            color,
        };
        let a = v(x0, y0, uv[0], uv[1]);
        let b = v(x0, y1, uv[0], uv[3]);
        let c = v(x1, y0, uv[2], uv[1]);
        let d = v(x1, y1, uv[2], uv[3]);
        self.emit(texture, [a, b, c]);
        self.emit(texture, [b, d, c]);
    }

    /// An axis-aligned solid rectangle (no anti-aliasing; for hairlines on the pixel grid).
    pub fn rect(&mut self, rect: Rect, color: Rgba) {
        self.textured(&UiTexture::Solid, rect, [0.0, 0.0, 1.0, 1.0], color);
    }

    /// The outline of a rounded rectangle grown by `grow` pixels (negative shrinks), clockwise
    /// from the top-left corner's start. Returns the point count.
    fn outline(
        rect: Rect,
        radius: f32,
        grow: f32,
        segments: usize,
        out: &mut [[f32; 2]; MAX_OUTLINE],
    ) -> usize {
        let r0 = radius.clamp(0.0, rect.w.min(rect.h) * 0.5);
        let r = (r0 + grow).max(0.0);
        let segments = segments.clamp(1, 12);
        // Corner centres (on the un-grown rect's inset by r0).
        let centres = [
            (rect.x + r0, rect.y + r0, std::f32::consts::PI),
            (rect.right() - r0, rect.y + r0, std::f32::consts::PI * 1.5),
            (rect.right() - r0, rect.bottom() - r0, 0.0),
            (rect.x + r0, rect.bottom() - r0, std::f32::consts::PI * 0.5),
        ];
        let mut n = 0;
        for (cx, cy, start) in centres {
            for i in 0..=segments {
                let a = start + std::f32::consts::FRAC_PI_2 * (i as f32 / segments as f32);
                out[n] = [cx + a.cos() * r, cy + a.sin() * r];
                n += 1;
            }
        }
        n
    }

    fn segments_for(radius: f32) -> usize {
        ((radius * 0.35).ceil() as usize).clamp(2, 12)
    }

    /// Fill a rounded rectangle with an anti-aliased edge.
    pub fn round_rect(&mut self, rect: Rect, radius: f32, fill: Fill) {
        if rect.w <= 0.0 || rect.h <= 0.0 {
            return;
        }
        let segments = Self::segments_for(radius);
        let mut inner = [[0.0; 2]; MAX_OUTLINE];
        let mut outer = [[0.0; 2]; MAX_OUTLINE];
        let n = Self::outline(rect, radius, -0.5, segments, &mut inner);
        Self::outline(rect, radius, 0.5, segments, &mut outer);
        let (cx, cy) = rect.center();
        let centre_color = fill.at([cx, cy]);
        for i in 0..n {
            let a = inner[i];
            let b = inner[(i + 1) % n];
            self.solid_tri([[cx, cy], a, b], [centre_color, fill.at(a), fill.at(b)]);
        }
        // Fringe: inner ring at full alpha to outer ring at zero.
        for i in 0..n {
            let j = (i + 1) % n;
            let (ia, ib, oa, ob) = (inner[i], inner[j], outer[i], outer[j]);
            let ca = fill.at(ia);
            let cb = fill.at(ib);
            let mut ta = fill.at(oa);
            let mut tb = fill.at(ob);
            ta[3] = 0.0;
            tb[3] = 0.0;
            self.solid_tri([ia, oa, ib], [ca, ta, cb]);
            self.solid_tri([ib, oa, ob], [cb, ta, tb]);
        }
    }

    /// A stroke of about `width` pixels centred `inset` pixels inside a rounded rectangle's
    /// edge (use `inset = width / 2` for a CSS border).
    pub fn round_rect_stroke(
        &mut self,
        rect: Rect,
        radius: f32,
        width: f32,
        inset: f32,
        color: Rgba,
    ) {
        if rect.w <= 0.0 || rect.h <= 0.0 || color[3] <= 0.0 {
            return;
        }
        let segments = Self::segments_for(radius);
        let half = (width * 0.5).max(0.5);
        let mut a = [[0.0; 2]; MAX_OUTLINE];
        let mut b = [[0.0; 2]; MAX_OUTLINE];
        let mut c = [[0.0; 2]; MAX_OUTLINE];
        let mut d = [[0.0; 2]; MAX_OUTLINE];
        // Profile across the stroke: 0 at ±(half + 0.5), full at ±(half - 0.5).
        let n = Self::outline(rect, radius, -inset + half + 0.5, segments, &mut a);
        Self::outline(
            rect,
            radius,
            -inset + (half - 0.5).max(0.0),
            segments,
            &mut b,
        );
        Self::outline(
            rect,
            radius,
            -inset - (half - 0.5).max(0.0),
            segments,
            &mut c,
        );
        Self::outline(rect, radius, -inset - half - 0.5, segments, &mut d);
        let strength = if width < 1.0 { width } else { 1.0 };
        let full = [color[0], color[1], color[2], color[3] * strength];
        let clear = [color[0], color[1], color[2], 0.0];
        self.ring(&a, &b, n, clear, full);
        if width > 1.0 {
            self.ring(&b, &c, n, full, full);
        }
        self.ring(&c, &d, n, full, clear);
    }

    fn ring(
        &mut self,
        outer: &[[f32; 2]; MAX_OUTLINE],
        inner: &[[f32; 2]; MAX_OUTLINE],
        n: usize,
        outer_color: Rgba,
        inner_color: Rgba,
    ) {
        for i in 0..n {
            let j = (i + 1) % n;
            self.solid_tri(
                [outer[i], inner[i], outer[j]],
                [outer_color, inner_color, outer_color],
            );
            self.solid_tri(
                [outer[j], inner[i], inner[j]],
                [outer_color, inner_color, inner_color],
            );
        }
    }

    /// A soft drop shadow **outside** a rounded rectangle (like CSS `box-shadow`, which is not
    /// drawn under the element): `color` at the edge fading to clear over `spread` pixels on
    /// each side `[left, top, right, bottom]`.
    pub fn round_rect_shadow(&mut self, rect: Rect, radius: f32, spread: [f32; 4], color: Rgba) {
        let segments = Self::segments_for(radius + spread[3]);
        let mut inner = [[0.0; 2]; MAX_OUTLINE];
        let mut outer = [[0.0; 2]; MAX_OUTLINE];
        let n = Self::outline(rect, radius, 0.0, segments, &mut inner);
        let grown = Rect::new(
            rect.x - spread[0],
            rect.y - spread[1],
            rect.w + spread[0] + spread[2],
            rect.h + spread[1] + spread[3],
        );
        let grow = spread.iter().copied().fold(0.0, f32::max);
        Self::outline(grown, radius + grow, 0.0, segments, &mut outer);
        let clear = [color[0], color[1], color[2], 0.0];
        // Two bands with an eased profile read much softer than one linear band.
        let mut mid = [[0.0; 2]; MAX_OUTLINE];
        for i in 0..n {
            mid[i] = [
                inner[i][0] + (outer[i][0] - inner[i][0]) * 0.35,
                inner[i][1] + (outer[i][1] - inner[i][1]) * 0.35,
            ];
        }
        let mid_color = [color[0], color[1], color[2], color[3] * 0.38];
        self.ring(&mid, &inner, n, mid_color, color);
        self.ring(&outer, &mid, n, clear, mid_color);
    }

    /// A filled, anti-aliased circle.
    pub fn circle(&mut self, cx: f32, cy: f32, r: f32, color: Rgba) {
        self.round_rect(
            Rect::new(cx - r, cy - r, r * 2.0, r * 2.0),
            r,
            Fill::Solid(color),
        );
    }

    /// An anti-aliased line segment `width` pixels wide.
    pub fn line(&mut self, x0: f32, y0: f32, x1: f32, y1: f32, width: f32, color: Rgba) {
        let (dx, dy) = (x1 - x0, y1 - y0);
        let len = (dx * dx + dy * dy).sqrt();
        if len <= 0.0 {
            return;
        }
        let (nx, ny) = (-dy / len, dx / len);
        let half = width * 0.5;
        let clear = [color[0], color[1], color[2], 0.0];
        let p = |o: f32| ([x0 + nx * o, y0 + ny * o], [x1 + nx * o, y1 + ny * o]);
        let (a0, a1) = p(-half - 0.5);
        let (b0, b1) = p(-half + 0.5);
        let (c0, c1) = p(half - 0.5);
        let (d0, d1) = p(half + 0.5);
        let core = if width < 1.0 {
            [color[0], color[1], color[2], color[3] * width]
        } else {
            color
        };
        let mut band =
            |p0: [f32; 2], p1: [f32; 2], q0: [f32; 2], q1: [f32; 2], cp: Rgba, cq: Rgba| {
                self.solid_tri([p0, q0, p1], [cp, cq, cp]);
                self.solid_tri([p1, q0, q1], [cp, cq, cq]);
            };
        band(a0, a1, b0, b1, clear, core);
        if width > 1.0 {
            band(b0, b1, c0, c1, core, core);
        }
        band(c0, c1, d0, d1, core, clear);
    }

    /// Fill `mesh` (a [`PathMesh`], built for its size) with `color`, its origin at `origin`.
    pub fn fill_mesh(&mut self, mesh: &PathMesh, origin: (f32, f32), color: Rgba) {
        for t in mesh.vertices.chunks_exact(3) {
            let at = |k: usize| [origin.0 + t[k].0[0], origin.1 + t[k].0[1]];
            let tint = |k: usize| [color[0], color[1], color[2], color[3] * t[k].1];
            self.solid_tri([at(0), at(1), at(2)], [tint(0), tint(1), tint(2)]);
        }
    }
}

/// A filled shape tessellated once for one size, for shapes the other primitives cannot make (a
/// logo): closed outlines in pixels around an origin, filled with the even-odd rule (an outline
/// inside another cuts a hole) and given the same one-pixel anti-aliasing ramp as every other
/// shape here, centred on the outline. Build it when the size changes; draw it each frame with
/// [`Painter::fill_mesh`], which only offsets and tints it.
///
/// The fill is the outlines moved half a pixel into the shape, cut into horizontal trapezoids at
/// every vertex and at every point where two edges cross (so the fill stays right where that
/// half-pixel move folds a sharp tip over itself); the ramp is a band from there to half a pixel
/// outside. Outlines must not cross each other or themselves.
#[derive(Debug, Clone, Default)]
pub struct PathMesh {
    /// A triangle list: position (relative to the origin) and coverage (1 inside, 0 at the outer
    /// edge of the ramp).
    vertices: Vec<([f32; 2], f32)>,
}

impl PathMesh {
    /// Points closer than this to the last point kept are dropped: it bounds the triangle count
    /// by the outline's length in pixels and keeps tiny steps from tilting the edge normals.
    const MIN_STEP: f32 = 0.5;

    /// Tessellate `outlines` (each a closed polygon in pixels; the last point joins the first).
    #[must_use]
    pub fn new(outlines: &[Vec<[f32; 2]>]) -> PathMesh {
        let contours: Vec<Vec<[f32; 2]>> = outlines
            .iter()
            .map(|outline| simplified(outline, Self::MIN_STEP))
            .filter(|contour| contour.len() >= 3)
            .collect();
        let mut inner = Vec::with_capacity(contours.len());
        let mut outer = Vec::with_capacity(contours.len());
        for (index, contour) in contours.iter().enumerate() {
            // Even-odd: an outline inside an odd number of others bounds a hole, so its filled
            // side is its outside.
            let probe = midpoint(contour[0], contour[1]);
            let depth = contours
                .iter()
                .enumerate()
                .filter(|&(other, c)| other != index && contains(c, probe))
                .count();
            let area: f32 = (0..contour.len())
                .map(|i| {
                    let (a, b) = (contour[i], contour[(i + 1) % contour.len()]);
                    a[0] * b[1] - b[0] * a[1]
                })
                .sum();
            // y down: for a positive shoelace sum (-dy, dx) points into the outline.
            let sign = if (area > 0.0) == (depth % 2 == 0) {
                1.0
            } else {
                -1.0
            };
            let (i, o) = offset_contour(contour, sign);
            inner.push(i);
            outer.push(o);
        }
        let mut mesh = PathMesh::default();
        mesh.fill_even_odd(&inner);
        for (i, o) in inner.iter().zip(&outer) {
            let n = i.len();
            for k in 0..n {
                let j = (k + 1) % n;
                mesh.push([(i[k], 1.0), (o[k], 0.0), (i[j], 1.0)]);
                mesh.push([(i[j], 1.0), (o[k], 0.0), (o[j], 0.0)]);
            }
        }
        mesh
    }

    /// Whether there is nothing to draw.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.vertices.is_empty()
    }

    /// How many triangles a draw emits.
    #[must_use]
    pub fn triangle_count(&self) -> usize {
        self.vertices.len() / 3
    }

    fn push(&mut self, triangle: [([f32; 2], f32); 3]) {
        let [a, b, c] = triangle.map(|v| v.0);
        let area = (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]);
        if area.abs() > 1e-6 {
            self.vertices.extend(triangle);
        }
    }

    /// Full-coverage trapezoids for the even-odd fill of `contours`.
    fn fill_even_odd(&mut self, contours: &[Vec<[f32; 2]>]) {
        let mut edges: Vec<([f32; 2], [f32; 2])> = Vec::new();
        for contour in contours {
            for k in 0..contour.len() {
                let (a, b) = (contour[k], contour[(k + 1) % contour.len()]);
                if (a[1] - b[1]).abs() > 1e-6 {
                    edges.push(if a[1] < b[1] { (a, b) } else { (b, a) });
                }
            }
        }
        // Slab boundaries: every vertex, and every point where two edges cross, so that inside a
        // slab no edges cross and pairing them left to right is exact.
        let mut ys: Vec<f32> = contours.iter().flatten().map(|p| p[1]).collect();
        for (n, &(a, b)) in edges.iter().enumerate() {
            for &(c, d) in &edges[n + 1..] {
                if let Some(y) = crossing_y(a, b, c, d) {
                    ys.push(y);
                }
            }
        }
        ys.sort_by(f32::total_cmp);
        ys.dedup_by(|y, previous| (*y - *previous).abs() < 1e-4);
        let x_at = |(a, b): ([f32; 2], [f32; 2]), y: f32| {
            a[0] + (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1])
        };
        let mut row: Vec<(f32, f32, f32)> = Vec::new();
        for slab in ys.windows(2) {
            let (y0, y1) = (slab[0], slab[1]);
            let mid = (y0 + y1) * 0.5;
            row.clear();
            row.extend(
                edges
                    .iter()
                    .filter(|(a, b)| a[1] < mid && mid < b[1])
                    .map(|&edge| (x_at(edge, mid), x_at(edge, y0), x_at(edge, y1))),
            );
            row.sort_by(|p, q| p.0.total_cmp(&q.0));
            for pair in row.chunks_exact(2) {
                let ((_, l0, l1), (_, r0, r1)) = (pair[0], pair[1]);
                self.push([([l0, y0], 1.0), ([r0, y0], 1.0), ([r1, y1], 1.0)]);
                self.push([([l0, y0], 1.0), ([r1, y1], 1.0), ([l1, y1], 1.0)]);
            }
        }
    }
}

fn midpoint(a: [f32; 2], b: [f32; 2]) -> [f32; 2] {
    [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5]
}

/// `outline` without its closing duplicate and without points closer than `step` to the last
/// one kept.
fn simplified(outline: &[[f32; 2]], step: f32) -> Vec<[f32; 2]> {
    let close = |a: [f32; 2], b: [f32; 2]| (a[0] - b[0]).hypot(a[1] - b[1]) < step;
    let mut kept: Vec<[f32; 2]> = Vec::with_capacity(outline.len());
    for &p in outline {
        if kept.last().is_none_or(|&last| !close(last, p)) {
            kept.push(p);
        }
    }
    while kept.len() > 1 && close(kept[0], kept[kept.len() - 1]) {
        kept.pop();
    }
    kept
}

/// Whether `p` is inside the polygon `contour` (even-odd crossing test).
fn contains(contour: &[[f32; 2]], p: [f32; 2]) -> bool {
    let mut inside = false;
    let n = contour.len();
    for k in 0..n {
        let (a, b) = (contour[k], contour[(k + n - 1) % n]);
        if (a[1] > p[1]) != (b[1] > p[1])
            && p[0] < a[0] + (b[0] - a[0]) * (p[1] - a[1]) / (b[1] - a[1])
        {
            inside = !inside;
        }
    }
    inside
}

/// The y where segments `ab` and `cd` cross, strictly inside both (so neighbours that only
/// share an end do not count).
fn crossing_y(a: [f32; 2], b: [f32; 2], c: [f32; 2], d: [f32; 2]) -> Option<f32> {
    let cross = |u: [f32; 2], v: [f32; 2]| u[0] * v[1] - u[1] * v[0];
    let (r, s) = ([b[0] - a[0], b[1] - a[1]], [d[0] - c[0], d[1] - c[1]]);
    let denominator = cross(r, s);
    if denominator.abs() < 1e-9 {
        return None;
    }
    let ac = [c[0] - a[0], c[1] - a[1]];
    let t = cross(ac, s) / denominator;
    let u = cross(ac, r) / denominator;
    let inside = |v: f32| v > 1e-5 && v < 1.0 - 1e-5;
    (inside(t) && inside(u)).then(|| a[1] + t * r[1])
}

/// `contour` moved half a pixel towards its filled side and half a pixel away from it: the
/// inner edge of the anti-aliasing ramp and its outer edge. `sign` is +1 when `(-dy, dx)` points
/// at the filled side. Mitred corners, the mitre capped at twice the offset.
fn offset_contour(contour: &[[f32; 2]], sign: f32) -> (Vec<[f32; 2]>, Vec<[f32; 2]>) {
    let n = contour.len();
    let normal = |k: usize| {
        let (a, b) = (contour[k], contour[(k + 1) % n]);
        let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
        let len = dx.hypot(dy).max(1e-6);
        [-dy / len * sign, dx / len * sign]
    };
    let mut inner = Vec::with_capacity(n);
    let mut outer = Vec::with_capacity(n);
    for (k, p) in contour.iter().enumerate() {
        let (before, after) = (normal((k + n - 1) % n), normal(k));
        let sum = [before[0] + after[0], before[1] + after[1]];
        let len = sum[0].hypot(sum[1]);
        let m = if len > 1e-4 {
            [sum[0] / len, sum[1] / len]
        } else {
            after
        };
        let reach = 0.5 / (m[0] * after[0] + m[1] * after[1]).max(0.5);
        inner.push([p[0] + m[0] * reach, p[1] + m[1] * reach]);
        outer.push([p[0] - m[0] * reach, p[1] - m[1] * reach]);
    }
    (inner, outer)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_rounded_rect_stays_inside_its_bounds_plus_the_fringe() {
        let mut list = DrawList::new();
        let mut p = Painter::new(&mut list);
        let rect = Rect::new(10.0, 20.0, 100.0, 50.0);
        p.round_rect(rect, 12.0, Fill::Solid(white(1.0)));
        assert!(!list.is_empty());
        for v in list.vertices() {
            assert!(v.pos[0] >= 9.4 && v.pos[0] <= 110.6, "{:?}", v.pos);
            assert!(v.pos[1] >= 19.4 && v.pos[1] <= 70.6, "{:?}", v.pos);
        }
        // The fringe fades to zero alpha.
        assert!(list.vertices().iter().any(|v| v.color[3] == 0.0));
        assert!(list.vertices().iter().any(|v| v.color[3] == 1.0));
    }

    #[test]
    fn clipping_cuts_triangles_at_the_clip_rectangle() {
        let mut list = DrawList::new();
        let mut p = Painter::new(&mut list);
        p.set_clip(Some(Rect::new(0.0, 0.0, 50.0, 50.0)));
        p.rect(Rect::new(25.0, 25.0, 100.0, 100.0), white(1.0));
        assert!(!list.is_empty());
        for v in list.vertices() {
            assert!(v.pos[0] <= 50.0001 && v.pos[1] <= 50.0001, "{:?}", v.pos);
        }
        // Fully outside: nothing at all.
        let mut list = DrawList::new();
        let mut p = Painter::new(&mut list);
        p.set_clip(Some(Rect::new(0.0, 0.0, 10.0, 10.0)));
        p.rect(Rect::new(20.0, 20.0, 5.0, 5.0), white(1.0));
        assert!(list.is_empty());
    }

    #[test]
    fn gradients_interpolate_along_the_css_angle() {
        let rect = Rect::new(0.0, 0.0, 100.0, 100.0);
        // 180deg: top to bottom.
        let fill = Fill::css_angle(rect, 180.0, white(1.0), black(1.0));
        let top = fill.at([50.0, 0.0]);
        let bottom = fill.at([50.0, 100.0]);
        assert!(top[0] > 0.99 && bottom[0] < 0.01, "{top:?} {bottom:?}");
        // 90deg: left to right.
        let fill = Fill::css_angle(rect, 90.0, white(1.0), black(1.0));
        assert!(fill.at([0.0, 50.0])[0] > 0.99 && fill.at([100.0, 50.0])[0] < 0.01);
    }

    /// The UI pipeline culls back faces by default; a clockwise fill fan vanished on the GPU
    /// (found in the resume run's first render: no switch knobs, dots or slider tracks).
    #[test]
    fn every_triangle_faces_the_culling_ui_pipeline() {
        let mut list = DrawList::new();
        let mut p = Painter::new(&mut list);
        let rect = Rect::new(10.0, 20.0, 100.0, 50.0);
        p.round_rect(rect, 12.0, Fill::Solid(white(1.0)));
        p.round_rect(
            rect,
            0.0,
            Fill::css_angle(rect, 135.0, white(1.0), black(1.0)),
        );
        p.round_rect_stroke(rect, 12.0, 1.0, 0.5, white(1.0));
        p.round_rect_shadow(rect, 12.0, [6.0; 4], black(0.3));
        p.circle(50.0, 50.0, 6.0, white(1.0));
        p.line(0.0, 0.0, 30.0, 10.0, 2.0, white(1.0));
        p.rect(Rect::new(1.0, 2.0, 3.0, 4.0), white(1.0));
        p.set_clip(Some(Rect::new(0.0, 0.0, 60.0, 40.0)));
        p.round_rect(rect, 12.0, Fill::Solid(white(1.0)));
        let vertices = list.vertices();
        assert!(!vertices.is_empty());
        assert_eq!(vertices.len() % 3, 0, "a triangle list");
        for t in vertices.chunks(3) {
            let cross = (t[1].pos[0] - t[0].pos[0]) * (t[2].pos[1] - t[0].pos[1])
                - (t[2].pos[0] - t[0].pos[0]) * (t[1].pos[1] - t[0].pos[1]);
            let corners: Vec<[f32; 2]> = t.iter().map(|v| v.pos).collect();
            assert!(
                cross <= 1e-3,
                "a clockwise triangle would be culled: {corners:?}"
            );
        }
    }

    /// A square with a square hole: covered inside, empty in the hole and outside, and every
    /// edge's ramp centred on the edge (half covered there, done half a pixel either side). The
    /// hole may wind either way: the rule is even-odd.
    #[test]
    fn a_path_fill_covers_the_inside_cuts_holes_and_ramps_on_the_edge() {
        use crate::widget::test_support::composite_at;
        let square =
            |x0: f32, y0: f32, x1: f32, y1: f32| vec![[x0, y0], [x1, y0], [x1, y1], [x0, y1]];
        for reversed_hole in [false, true] {
            let mut hole = square(6.0, 6.0, 14.0, 14.0);
            if reversed_hole {
                hole.reverse();
            }
            let mesh = PathMesh::new(&[square(0.0, 0.0, 20.0, 20.0), hole]);
            let mut list = DrawList::new();
            Painter::new(&mut list).fill_mesh(&mesh, (100.0, 50.0), white(1.0));
            let cover = |x: f32, y: f32| composite_at(&list, (100.0 + x, 50.0 + y), [0.0; 3])[0];
            for (x, y, expected, what) in [
                (3.37, 10.29, 1.0, "inside"),
                (17.21, 3.13, 1.0, "inside, near a corner"),
                (10.37, 10.29, 0.0, "the hole"),
                (-3.0, 10.29, 0.0, "outside"),
                (0.0, 10.29, 0.5, "on the outer edge"),
                (-0.52, 10.29, 0.0, "past the ramp"),
                (0.52, 10.29, 1.0, "inside the ramp"),
                (6.0, 9.71, 0.5, "on the hole's edge"),
                (6.52, 9.71, 0.0, "inside the hole, past the ramp"),
                (5.48, 9.71, 1.0, "before the hole's ramp"),
            ] {
                let got = cover(x, y);
                assert!(
                    (got - expected).abs() < 0.05,
                    "{what} ({x}, {y}): {got}, reversed hole {reversed_hole}"
                );
            }
        }
    }

    /// Curved outlines (a ring of two circles) come out round and hole-cut, with a bounded
    /// triangle count, and the painter's alpha, offset and culling-safe winding apply.
    #[test]
    fn a_path_fill_goes_through_the_painters_alpha_offset_and_winding() {
        use crate::widget::test_support::composite_at;
        let circle = |r: f32| -> Vec<[f32; 2]> {
            (0..96)
                .map(|i| {
                    let a = i as f32 / 96.0 * std::f32::consts::TAU;
                    [a.cos() * r, a.sin() * r]
                })
                .collect()
        };
        let mesh = PathMesh::new(&[circle(12.0), circle(6.0)]);
        assert!(!mesh.is_empty());
        assert!(mesh.triangle_count() < 900, "{}", mesh.triangle_count());
        let mut list = DrawList::new();
        let mut p = Painter::new(&mut list);
        p.alpha = 0.5;
        p.offset = (40.0, 30.0);
        p.fill_mesh(&mesh, (10.0, 20.0), white(1.0));
        let (cx, cy) = (50.0, 50.0);
        let at = |dx: f32, dy: f32| composite_at(&list, (cx + dx, cy + dy), [0.0; 3])[0];
        assert!(
            (at(9.03, 0.31) - 0.5).abs() < 0.03,
            "in the ring: {}",
            at(9.03, 0.31)
        );
        assert!(at(0.31, 0.27) < 0.01, "the hole");
        assert!(at(-6.31, 6.43) > 0.45, "in the ring, diagonally");
        assert!(at(14.0, 0.3) < 0.01, "outside");
        for t in list.vertices().chunks(3) {
            assert!(t.iter().all(|v| v.color[3] <= 0.5 + 1e-6));
            let cross = (t[1].pos[0] - t[0].pos[0]) * (t[2].pos[1] - t[0].pos[1])
                - (t[2].pos[0] - t[0].pos[0]) * (t[1].pos[1] - t[0].pos[1]);
            assert!(cross <= 1e-3, "a clockwise triangle would be culled");
        }
        // Degenerate input draws nothing rather than panicking.
        assert!(PathMesh::new(&[vec![[0.0, 0.0], [0.1, 0.1]]]).is_empty());
        assert!(PathMesh::new(&[]).is_empty());
    }

    #[test]
    fn alpha_and_offset_apply_to_everything() {
        let mut list = DrawList::new();
        let mut p = Painter::new(&mut list);
        p.alpha = 0.5;
        p.offset = (3.0, 4.0);
        p.rect(Rect::new(0.0, 0.0, 2.0, 2.0), white(0.8));
        assert!(list
            .vertices()
            .iter()
            .all(|v| (v.color[3] - 0.4).abs() < 1e-6 && v.pos[0] >= 3.0 && v.pos[1] >= 4.0));
    }
}
