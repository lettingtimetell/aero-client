//! The title screen's button icons: clean line icons drawn with the mod menu's [`Painter`], on a
//! 24-unit grid centred in the icon box (the usual icon-font grid, so the shapes read at any
//! size). Strokes have round joins and caps: every segment is an anti-aliased line, with a dot
//! of the stroke's width on each joint.
//!
//! The four shapes are drawn here from scratch: two people (Multiplayer), a cog (Options), a
//! globe (Language) and the power symbol (Quit Game).

use std::f32::consts::{PI, TAU};

use crate::screens::modmenu::paint::{Painter, Rect, Rgba};

/// Which icon.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Icon {
    /// Two people: playing with others.
    Multiplayer,
    /// A cog.
    Options,
    /// A globe.
    Language,
    /// The power symbol.
    Quit,
    /// A ring arrow turning back round a small play triangle: Replays.
    Replays,
    /// Three sliders: Mods.
    Mods,
}

/// Maps the 24-unit grid to pixels and strokes paths.
struct Pen<'p, 'a> {
    painter: &'p mut Painter<'a>,
    cx: f32,
    cy: f32,
    k: f32,
    width: f32,
    color: Rgba,
}

impl Pen<'_, '_> {
    fn at(&self, x: f32, y: f32) -> [f32; 2] {
        [self.cx + (x - 12.0) * self.k, self.cy + (y - 12.0) * self.k]
    }

    /// A polyline in grid units, round-joined; `closed` joins the last point to the first.
    fn path(&mut self, points: &[[f32; 2]], closed: bool) {
        if points.len() < 2 {
            return;
        }
        let px: Vec<[f32; 2]> = points.iter().map(|p| self.at(p[0], p[1])).collect();
        let n = px.len();
        let segments = if closed { n } else { n - 1 };
        for i in 0..segments {
            let (a, b) = (px[i], px[(i + 1) % n]);
            self.painter
                .line(a[0], a[1], b[0], b[1], self.width, self.color);
        }
        // Round caps on an open path's ends, and round joins where the path turns enough to
        // open a visible notch (an arc's gentle steps do not: that keeps arcs cheap).
        let r = self.width * 0.5;
        let direction = |a: [f32; 2], b: [f32; 2]| (b[1] - a[1]).atan2(b[0] - a[0]);
        for i in 0..n {
            let end = !closed && (i == 0 || i == n - 1);
            let sharp = if closed || (i > 0 && i < n - 1) {
                let before = direction(px[(i + n - 1) % n], px[i]);
                let after = direction(px[i], px[(i + 1) % n]);
                let turn = (after - before + PI).rem_euclid(TAU) - PI;
                turn.abs() > 0.4
            } else {
                false
            };
            if end || sharp {
                self.painter.circle(px[i][0], px[i][1], r, self.color);
            }
        }
    }

    /// An elliptical arc in grid units from angle `a0` to `a1` (radians, y down: 0 is right,
    /// `PI / 2` is down).
    fn arc(&mut self, c: [f32; 2], r: [f32; 2], a0: f32, a1: f32) {
        let steps = (((a1 - a0).abs() / TAU) * 40.0).ceil().max(2.0) as usize;
        let points: Vec<[f32; 2]> = (0..=steps)
            .map(|i| {
                let a = a0 + (a1 - a0) * i as f32 / steps as f32;
                [c[0] + a.cos() * r[0], c[1] + a.sin() * r[1]]
            })
            .collect();
        self.path(&points, false);
    }

    /// A circle outline in grid units.
    fn circle(&mut self, c: [f32; 2], r: f32) {
        let centre = self.at(c[0], c[1]);
        let r = r * self.k;
        self.painter.round_rect_stroke(
            Rect::new(centre[0] - r, centre[1] - r, r * 2.0, r * 2.0),
            r,
            self.width,
            0.0,
            self.color,
        );
    }
}

/// Draw `icon` centred on `(cx, cy)`, `size` pixels across, with strokes `stroke` pixels wide.
pub fn draw(
    painter: &mut Painter<'_>,
    icon: Icon,
    (cx, cy): (f32, f32),
    size: f32,
    stroke: f32,
    color: Rgba,
) {
    let mut pen = Pen {
        painter,
        cx,
        cy,
        k: size / 24.0,
        width: stroke,
        color,
    };
    match icon {
        Icon::Multiplayer => {
            // The front person: head and shoulders.
            pen.circle([9.0, 8.0], 3.6);
            pen.arc([9.0, 20.0], [6.6, 6.0], PI, TAU);
            // The person behind, to the right: the visible half of the head and one shoulder.
            pen.arc([15.6, 7.6], [3.3, 3.3], -PI * 0.42, PI * 0.5);
            pen.arc([16.4, 20.0], [5.0, 5.8], -PI * 0.5, 0.0);
        }
        Icon::Options => {
            // A cog: eight teeth around a ring, and the hub.
            let (outer, inner) = (10.2, 7.4);
            let mut outline = Vec::with_capacity(8 * 4);
            for tooth in 0..8 {
                let a = tooth as f32 * TAU / 8.0;
                let deg = PI / 180.0;
                for (r, da) in [
                    (inner, -15.0 * deg),
                    (outer, -9.0 * deg),
                    (outer, 9.0 * deg),
                    (inner, 15.0 * deg),
                ] {
                    outline.push([12.0 + (a + da).cos() * r, 12.0 + (a + da).sin() * r]);
                }
            }
            pen.path(&outline, true);
            pen.circle([12.0, 12.0], 3.2);
        }
        Icon::Language => {
            // A globe: the outline, one meridian and the equator.
            pen.circle([12.0, 12.0], 9.6);
            pen.arc([12.0, 12.0], [4.2, 9.6], 0.0, TAU);
            pen.path(&[[2.4, 12.0], [21.6, 12.0]], false);
        }
        Icon::Quit => {
            // The power symbol: a ring open at the top, and the bar through the gap.
            pen.arc([12.0, 13.0], [8.4, 8.4], -PI * 0.5 + 0.62, PI * 1.5 - 0.62);
            pen.path(&[[12.0, 2.6], [12.0, 11.4]], false);
        }
        Icon::Replays => {
            // A ring open at the top left, its arrowhead turning back (counter-clockwise, as a
            // replay arrow does), round a small outlined play triangle.
            let (a0, a1) = (-PI * 0.62, PI * 1.15);
            pen.arc([12.0, 12.0], [9.2, 9.2], a0, a1);
            let tip = [12.0 + 9.2 * a0.cos(), 12.0 + 9.2 * a0.sin()];
            // y down: decreasing angles run counter-clockwise; the head points along them.
            let along = a0 - PI * 0.5;
            let back = |da: f32| {
                [
                    tip[0] - (along + da).cos() * 3.6,
                    tip[1] - (along + da).sin() * 3.6,
                ]
            };
            pen.path(&[back(0.62), tip, back(-0.62)], false);
            pen.path(&[[10.2, 8.6], [15.6, 12.0], [10.2, 15.4]], true);
        }
        Icon::Mods => {
            // Three sliders, each knob at another place: settings to tune.
            for (y, knob) in [(5.6, 8.4), (12.0, 15.6), (18.4, 10.6)] {
                let gap = 2.9;
                if knob - gap > 3.0 {
                    pen.path(&[[3.0, y], [knob - gap, y]], false);
                }
                if knob + gap < 21.0 {
                    pen.path(&[[knob + gap, y], [21.0, y]], false);
                }
                pen.circle([knob, y], 2.4);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::widget::DrawList;

    #[test]
    fn every_icon_draws_inside_its_box() {
        for icon in [
            Icon::Multiplayer,
            Icon::Options,
            Icon::Language,
            Icon::Quit,
            Icon::Replays,
            Icon::Mods,
        ] {
            let mut list = DrawList::new();
            let mut painter = Painter::new(&mut list);
            draw(&mut painter, icon, (50.0, 50.0), 24.0, 2.0, [1.0; 4]);
            assert!(!list.is_empty(), "{icon:?} drew nothing");
            for v in list.vertices() {
                // Half the box plus the stroke and its anti-aliasing fringe.
                assert!(
                    (v.pos[0] - 50.0).abs() <= 13.6 && (v.pos[1] - 50.0).abs() <= 13.6,
                    "{icon:?} leaves its box: {:?}",
                    v.pos
                );
            }
        }
    }

    #[test]
    fn icons_scale_with_their_size() {
        let extent = |size: f32| {
            let mut list = DrawList::new();
            let mut painter = Painter::new(&mut list);
            draw(
                &mut painter,
                Icon::Language,
                (0.0, 0.0),
                size,
                1.0,
                [1.0; 4],
            );
            list.vertices()
                .iter()
                .map(|v| v.pos[0].abs())
                .fold(0.0_f32, f32::max)
        };
        let (small, large) = (extent(24.0), extent(48.0));
        assert!((large - 2.0 * small).abs() < 1.5, "{small} {large}");
    }
}
