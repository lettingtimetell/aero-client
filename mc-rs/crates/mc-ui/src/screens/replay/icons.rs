//! The replay screens' icons: clean line icons on a 24-unit grid centred in the icon box (the
//! title screen's icon style, `aero_title::icons`), plus a few filled marks (play, pause, step)
//! where a video player's controls are filled by convention. Drawn with the mod menu's
//! [`Painter`] and [`super::style::fill_convex`].

use std::f32::consts::{PI, TAU};

use super::style::fill_convex;
use crate::screens::modmenu::paint::{Fill, Painter, Rect, Rgba};

/// Which icon.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Icon {
    /// A filled play triangle.
    Play,
    /// Two filled bars.
    Pause,
    /// A counter-clockwise arrow: back 5 seconds.
    Back5,
    /// A clockwise arrow: forward 5 seconds.
    Forward5,
    /// A bar and a triangle pointing left: one tick back.
    StepBack,
    /// A triangle and a bar pointing right: one tick forward.
    StepForward,
    /// An eye: the recorded first-person view.
    CameraPov,
    /// A person seen from behind and above: third-person follow.
    CameraFollow,
    /// Four arrows: the free camera.
    CameraFree,
    /// A left bracket with a dot: set the selection's In point.
    MarkIn,
    /// A right bracket with a dot: set the Out point.
    MarkOut,
    /// Two arrows chasing round: preview the selection on a loop.
    Loop,
    /// Three sliders: render settings.
    Sliders,
    /// An arrow out of a tray: export.
    Export,
    /// An eye struck through: hide the controls.
    Hide,
    /// An arrow leaving a door frame: exit.
    Exit,
    /// A strip of film: a recording.
    Film,
    /// Scissors: a clip.
    Scissors,
    /// A pencil: rename.
    Pencil,
    /// A folder.
    Folder,
    /// A trash can.
    Trash,
    /// A cross.
    Close,
    /// An arrow pointing left.
    Back,
    /// A tick.
    Check,
    /// A circled exclamation mark.
    Warning,
    /// A microphone.
    Mic,
    /// A speaker.
    Speaker,
    /// The title screen's Replays button: a circular arrow round a play triangle.
    Replay,
    /// A small clock.
    Clock,
    /// A circle with a dot: the recording.
    Record,
    /// A cog: the game's options (video, chat, controls...).
    Gear,
    /// An arrow pointing up: up one folder.
    Up,
    /// Three lines getting shorter beside a down arrow: the sort order.
    Sort,
    /// Three dotted lines: every replay in one list.
    List,
}

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

    /// A polyline, round-joined and round-capped (a dot on every joint).
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
        let r = self.width * 0.5;
        if r >= 0.9 {
            for p in &px {
                self.painter.circle(p[0], p[1], r, self.color);
            }
        }
    }

    fn arc(&mut self, c: [f32; 2], r: f32, a0: f32, a1: f32) {
        let steps = (((a1 - a0).abs() / TAU) * 36.0).ceil().max(2.0) as usize;
        let points: Vec<[f32; 2]> = (0..=steps)
            .map(|i| {
                let a = a0 + (a1 - a0) * i as f32 / steps as f32;
                [c[0] + a.cos() * r, c[1] + a.sin() * r]
            })
            .collect();
        // Only the ends get round caps (the arc's gentle joints need none).
        let px: Vec<[f32; 2]> = points.iter().map(|p| self.at(p[0], p[1])).collect();
        for pair in px.windows(2) {
            self.painter.line(
                pair[0][0], pair[0][1], pair[1][0], pair[1][1], self.width, self.color,
            );
        }
        let r = self.width * 0.5;
        if r >= 0.9 {
            for p in [px[0], px[px.len() - 1]] {
                self.painter.circle(p[0], p[1], r, self.color);
            }
        }
    }

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

    fn dot(&mut self, c: [f32; 2], r: f32) {
        let p = self.at(c[0], c[1]);
        self.painter.circle(p[0], p[1], r * self.k, self.color);
    }

    fn fill(&mut self, points: &[[f32; 2]]) {
        let px: Vec<[f32; 2]> = points.iter().map(|p| self.at(p[0], p[1])).collect();
        fill_convex(self.painter, &px, self.color);
    }

    fn bar(&mut self, x: f32, y: f32, w: f32, h: f32, r: f32) {
        let p = self.at(x, y);
        self.painter.round_rect(
            Rect::new(p[0], p[1], w * self.k, h * self.k),
            r * self.k,
            Fill::Solid(self.color),
        );
    }

    fn rect_outline(&mut self, x: f32, y: f32, w: f32, h: f32, r: f32) {
        let p = self.at(x, y);
        self.painter.round_rect_stroke(
            Rect::new(p[0], p[1], w * self.k, h * self.k),
            r * self.k,
            self.width,
            0.0,
            self.color,
        );
    }

    /// An arrowhead at `tip` pointing along `angle` (radians, y down).
    fn head(&mut self, tip: [f32; 2], angle: f32, size: f32) {
        let back = |da: f32| {
            [
                tip[0] - (angle + da).cos() * size,
                tip[1] - (angle + da).sin() * size,
            ]
        };
        self.path(&[back(0.6), tip, back(-0.6)], false);
    }
}

/// Draw `icon` centred on `(cx, cy)`, `size` pixels across, strokes `stroke` pixels wide.
#[allow(clippy::too_many_lines)]
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
        Icon::Play => pen.fill(&[[7.0, 4.5], [19.5, 12.0], [7.0, 19.5]]),
        Icon::Pause => {
            pen.bar(6.0, 4.5, 4.2, 15.0, 1.2);
            pen.bar(13.8, 4.5, 4.2, 15.0, 1.2);
        }
        Icon::Back5 | Icon::Forward5 => {
            let back = icon == Icon::Back5;
            // An open ring with an arrowhead; the "5" is drawn as strokes in the middle.
            let (a0, a1) = if back {
                (-PI * 0.5, PI * 1.25)
            } else {
                (-PI * 0.25, PI * 1.5)
            };
            pen.arc([12.0, 12.5], 8.6, a0, a1);
            if back {
                let tip = [12.0 + 8.6 * a0.cos(), 12.5 + 8.6 * a0.sin()];
                pen.head(tip, PI, 3.2);
            } else {
                let tip = [12.0 + 8.6 * a1.cos(), 12.5 + 8.6 * a1.sin()];
                pen.head(tip, 0.0, 3.2);
            }
            // "5", a little thinner than the ring.
            pen.width *= 0.8;
            pen.path(
                &[
                    [14.2, 8.8],
                    [10.6, 8.8],
                    [10.2, 12.0],
                    [12.4, 11.6],
                    [14.0, 12.6],
                    [14.0, 14.6],
                    [12.4, 15.8],
                    [10.2, 15.2],
                ],
                false,
            );
        }
        Icon::StepBack => {
            pen.bar(5.0, 5.5, 2.8, 13.0, 1.0);
            pen.fill(&[[19.0, 5.5], [19.0, 18.5], [9.0, 12.0]]);
        }
        Icon::StepForward => {
            pen.fill(&[[5.0, 5.5], [15.0, 12.0], [5.0, 18.5]]);
            pen.bar(16.2, 5.5, 2.8, 13.0, 1.0);
        }
        Icon::CameraPov | Icon::Hide => {
            // An eye: two arcs meeting at (3, 12) and (21, 12), 6 units tall each, and the pupil.
            pen.arc([12.0, 15.75], 9.75, -PI * 0.874, -PI * 0.126);
            pen.arc([12.0, 8.25], 9.75, PI * 0.126, PI * 0.874);
            pen.circle([12.0, 12.0], 3.0);
            if icon == Icon::Hide {
                pen.path(&[[4.0, 20.0], [20.0, 4.0]], false);
            }
        }
        Icon::CameraFollow => {
            // A person (head and shoulders) with a small camera behind, up and to the left.
            pen.circle([14.0, 9.0], 3.2);
            pen.arc([14.0, 21.0], 6.4, PI * 1.05, PI * 1.95);
            pen.rect_outline(2.5, 3.0, 6.0, 4.6, 1.0);
            pen.path(&[[8.5, 4.3], [10.2, 3.3], [10.2, 7.3], [8.5, 6.3]], false);
        }
        Icon::CameraFree => {
            pen.path(&[[12.0, 3.5], [12.0, 20.5]], false);
            pen.path(&[[3.5, 12.0], [20.5, 12.0]], false);
            pen.head([12.0, 3.5], -PI * 0.5, 3.0);
            pen.head([12.0, 20.5], PI * 0.5, 3.0);
            pen.head([3.5, 12.0], PI, 3.0);
            pen.head([20.5, 12.0], 0.0, 3.0);
        }
        Icon::MarkIn => {
            pen.path(&[[10.0, 4.0], [5.5, 4.0], [5.5, 20.0], [10.0, 20.0]], false);
            pen.fill(&[[11.0, 7.0], [19.0, 12.0], [11.0, 17.0]]);
        }
        Icon::MarkOut => {
            pen.path(
                &[[14.0, 4.0], [18.5, 4.0], [18.5, 20.0], [14.0, 20.0]],
                false,
            );
            pen.fill(&[[5.0, 7.0], [13.0, 12.0], [5.0, 17.0]]);
        }
        Icon::Loop => {
            pen.arc([12.0, 12.0], 7.5, -PI * 0.95, -PI * 0.05);
            pen.head(
                [
                    12.0 + 7.5 * (-PI * 0.05).cos(),
                    12.0 + 7.5 * (-PI * 0.05).sin(),
                ],
                PI * 0.45,
                3.0,
            );
            pen.arc([12.0, 12.0], 7.5, PI * 0.05, PI * 0.95);
            pen.head(
                [
                    12.0 + 7.5 * (PI * 0.95).cos(),
                    12.0 + 7.5 * (PI * 0.95).sin(),
                ],
                -PI * 0.55,
                3.0,
            );
        }
        Icon::Sliders => {
            for (y, knob) in [(6.0, 15.0), (12.0, 8.0), (18.0, 13.0)] {
                pen.path(&[[3.5, y], [20.5, y]], false);
                pen.dot([knob, y], 2.6);
            }
        }
        Icon::Export => {
            pen.path(&[[12.0, 3.5], [12.0, 14.5]], false);
            pen.head([12.0, 3.5], -PI * 0.5, 4.2);
            pen.path(
                &[
                    [7.5, 10.0],
                    [4.5, 10.0],
                    [4.5, 20.0],
                    [19.5, 20.0],
                    [19.5, 10.0],
                    [16.5, 10.0],
                ],
                false,
            );
        }
        Icon::Exit => {
            pen.path(&[[11.0, 4.0], [4.5, 4.0], [4.5, 20.0], [11.0, 20.0]], false);
            pen.path(&[[9.5, 12.0], [20.5, 12.0]], false);
            pen.head([20.5, 12.0], 0.0, 4.0);
        }
        Icon::Film => {
            pen.rect_outline(3.5, 5.0, 17.0, 14.0, 2.0);
            pen.path(&[[7.5, 5.0], [7.5, 19.0]], false);
            pen.path(&[[16.5, 5.0], [16.5, 19.0]], false);
            for y in [8.5, 12.0, 15.5] {
                pen.dot([5.5, y], 0.8);
                pen.dot([18.5, y], 0.8);
            }
        }
        Icon::Scissors => {
            pen.circle([6.5, 17.0], 2.8);
            pen.circle([17.5, 17.0], 2.8);
            pen.path(&[[8.6, 15.1], [17.0, 3.5]], false);
            pen.path(&[[15.4, 15.1], [7.0, 3.5]], false);
        }
        Icon::Pencil => {
            pen.path(
                &[
                    [4.5, 19.5],
                    [5.5, 15.0],
                    [15.5, 5.0],
                    [19.0, 8.5],
                    [9.0, 18.5],
                    [4.5, 19.5],
                ],
                false,
            );
            pen.path(&[[13.5, 7.0], [17.0, 10.5]], false);
        }
        Icon::Folder => {
            pen.path(
                &[
                    [3.5, 7.0],
                    [3.5, 18.5],
                    [20.5, 18.5],
                    [20.5, 8.5],
                    [11.5, 8.5],
                    [9.5, 5.5],
                    [3.5, 5.5],
                    [3.5, 7.0],
                ],
                false,
            );
        }
        Icon::Trash => {
            pen.path(&[[4.0, 6.5], [20.0, 6.5]], false);
            pen.path(&[[9.5, 6.5], [9.5, 4.0], [14.5, 4.0], [14.5, 6.5]], false);
            pen.path(&[[6.0, 6.5], [7.0, 20.0], [17.0, 20.0], [18.0, 6.5]], false);
            pen.path(&[[10.0, 10.0], [10.2, 16.5]], false);
            pen.path(&[[14.0, 10.0], [13.8, 16.5]], false);
        }
        Icon::Close => {
            pen.path(&[[6.0, 6.0], [18.0, 18.0]], false);
            pen.path(&[[18.0, 6.0], [6.0, 18.0]], false);
        }
        Icon::Back => {
            pen.path(&[[19.5, 12.0], [4.5, 12.0]], false);
            pen.head([4.5, 12.0], PI, 6.0);
        }
        Icon::Check => pen.path(&[[4.5, 12.5], [9.5, 17.5], [19.5, 6.5]], false),
        Icon::Warning => {
            pen.circle([12.0, 12.0], 9.0);
            pen.path(&[[12.0, 7.0], [12.0, 13.0]], false);
            pen.dot([12.0, 16.6], 1.3);
        }
        Icon::Mic => {
            pen.rect_outline(9.0, 3.0, 6.0, 11.0, 3.0);
            pen.arc([12.0, 10.5], 6.5, PI * 0.0, PI * 1.0);
            pen.path(&[[12.0, 17.0], [12.0, 21.0]], false);
        }
        Icon::Speaker => {
            pen.path(
                &[
                    [4.0, 9.5],
                    [8.0, 9.5],
                    [13.0, 5.0],
                    [13.0, 19.0],
                    [8.0, 14.5],
                    [4.0, 14.5],
                    [4.0, 9.5],
                ],
                false,
            );
            pen.arc([13.0, 12.0], 4.0, -PI * 0.3, PI * 0.3);
            pen.arc([13.0, 12.0], 7.5, -PI * 0.3, PI * 0.3);
        }
        Icon::Replay => {
            // A ring open at the top left with an arrowhead, round a small play triangle.
            let (a0, a1) = (-PI * 0.62, PI * 1.15);
            pen.arc([12.0, 12.0], 9.4, a0, a1);
            let tip = [12.0 + 9.4 * a0.cos(), 12.0 + 9.4 * a0.sin()];
            // Counter-clockwise (y down: decreasing angle), as a replay arrow turns.
            pen.head(tip, a0 - PI * 0.5, 3.4);
            pen.fill(&[[10.0, 8.2], [16.0, 12.0], [10.0, 15.8]]);
        }
        Icon::Clock => {
            pen.circle([12.0, 12.0], 8.5);
            pen.path(&[[12.0, 7.0], [12.0, 12.0], [15.5, 14.0]], false);
        }
        Icon::Record => {
            pen.circle([12.0, 12.0], 8.5);
            pen.dot([12.0, 12.0], 4.2);
        }
        Icon::Gear => {
            // A cog: a ring with eight teeth, and the hub.
            pen.circle([12.0, 12.0], 6.0);
            pen.circle([12.0, 12.0], 2.3);
            pen.width *= 1.7;
            for i in 0..8 {
                let (s, c) = (i as f32 * TAU / 8.0).sin_cos();
                pen.path(
                    &[
                        [12.0 + c * 7.3, 12.0 + s * 7.3],
                        [12.0 + c * 9.0, 12.0 + s * 9.0],
                    ],
                    false,
                );
            }
        }
        // The browser's folders and sort orders.
        Icon::Up => {
            pen.path(&[[12.0, 19.5], [12.0, 4.5]], false);
            pen.head([12.0, 4.5], -PI * 0.5, 6.0);
        }
        Icon::Sort => {
            pen.path(&[[3.5, 6.0], [13.0, 6.0]], false);
            pen.path(&[[3.5, 12.0], [10.5, 12.0]], false);
            pen.path(&[[3.5, 18.0], [8.0, 18.0]], false);
            pen.path(&[[18.0, 4.5], [18.0, 19.5]], false);
            pen.head([18.0, 19.5], PI * 0.5, 3.6);
        }
        Icon::List => {
            for y in [6.0, 12.0, 18.0] {
                pen.dot([5.0, y], 1.5);
                pen.path(&[[9.0, y], [20.5, y]], false);
            }
        }
    }
}

/// Every icon (tests and the gallery).
pub const ALL: [Icon; 34] = [
    Icon::Play,
    Icon::Pause,
    Icon::Back5,
    Icon::Forward5,
    Icon::StepBack,
    Icon::StepForward,
    Icon::CameraPov,
    Icon::CameraFollow,
    Icon::CameraFree,
    Icon::MarkIn,
    Icon::MarkOut,
    Icon::Loop,
    Icon::Sliders,
    Icon::Export,
    Icon::Hide,
    Icon::Exit,
    Icon::Film,
    Icon::Scissors,
    Icon::Pencil,
    Icon::Folder,
    Icon::Trash,
    Icon::Close,
    Icon::Back,
    Icon::Check,
    Icon::Warning,
    Icon::Mic,
    Icon::Speaker,
    Icon::Replay,
    Icon::Clock,
    Icon::Record,
    Icon::Gear,
    Icon::Up,
    Icon::Sort,
    Icon::List,
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::widget::DrawList;

    #[test]
    fn every_icon_draws_inside_its_box() {
        for icon in ALL {
            let mut list = DrawList::new();
            let mut painter = Painter::new(&mut list);
            draw(&mut painter, icon, (50.0, 50.0), 24.0, 2.0, [1.0; 4]);
            assert!(!list.is_empty(), "{icon:?} drew nothing");
            for v in list.vertices() {
                assert!(
                    (v.pos[0] - 50.0).abs() <= 13.6 && (v.pos[1] - 50.0).abs() <= 13.6,
                    "{icon:?} leaves its box: {:?}",
                    v.pos
                );
            }
        }
    }
}
