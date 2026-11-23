//! Window 1, "Where": the search field, then My setup, Keybinds and
//! Profiles, the categories, and All mods. While searching, each entry shows how many results it
//! holds and a click narrows the results to it.

use mc_mods::menu_layout::CATEGORIES;

use super::layout::{draw_title, Area};
use super::paint::{black, css, white, Fill, Painter, Rect};
use super::state::{NavRow, Pane, Target, View};
use super::text::{TextStyle, Weight};
use super::widgets::{draw_focus_ring, ms_since, shadowed};
use super::{MenuEnv, ModMenuScreen};

/// The search field's height.
const SEARCH_H: f32 = 36.0;
/// A window-1 entry.
const ENTRY_H: f32 = 32.0;
const ENTRY_GAP: f32 = 2.0;
/// The space a divider line takes.
const DIVIDER_H: f32 = 13.0;

/// The number (and warning mark) at the right of a window-1 entry.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct NavBadge {
    /// The count shown at the right, if any.
    pub(super) count: Option<usize>,
    /// A warning mark before the count (Keybinds: two keys clash).
    pub(super) warn: bool,
}

/// A thin line between groups of entries.
fn draw_divider(painter: &mut Painter<'_>, area: &Area, y: f32) -> f32 {
    let s = area.s;
    let mid = (y + DIVIDER_H * s * 0.5).round();
    painter.rect(
        Rect::new(area.x + 10.0 * s, mid, area.w - 20.0 * s, 1.0),
        white(0.14),
    );
    y + DIVIDER_H * s
}

impl ModMenuScreen {
    /// Window 1 at `rect`.
    pub(super) fn draw_sidebar(
        &mut self,
        painter: &mut Painter<'_>,
        env: &mut MenuEnv<'_>,
        rect: Rect,
        s: f32,
    ) {
        draw_title(painter, env, rect, "Aero", "Mods and settings", 0.0, s);
        let area = self.begin_body(painter, rect, Pane::Sidebar, s);
        let field = Rect::new(area.x, area.y, area.w, SEARCH_H * s);
        self.draw_search_field(painter, env, field, &area);
        let mut y = field.bottom() + 10.0 * s;
        let settings = env.settings;
        let searching = self.searching();
        let active = if searching {
            self.narrow.unwrap_or(View::AllMods)
        } else {
            self.view.unwrap_or(View::MySetup)
        };
        let setup = NavBadge {
            count: Some(if searching {
                self.found_count_in(settings, View::MySetup)
            } else {
                self.my_setup_len(settings)
            }),
            warn: false,
        };
        y = self.draw_entry(
            painter,
            env,
            View::MySetup,
            "My setup",
            setup,
            &area,
            y,
            active,
        );
        // The Keybinds and Profiles badges come from their files.
        let keys = if searching {
            NavBadge::default()
        } else {
            self.keybinds_badge(env)
        };
        y = self.draw_entry(
            painter,
            env,
            View::Keybinds,
            "Keybinds",
            keys,
            &area,
            y,
            active,
        );
        let profiles = if searching {
            NavBadge::default()
        } else {
            self.profiles_badge(env)
        };
        y = self.draw_entry(
            painter,
            env,
            View::Profiles,
            "Profiles",
            profiles,
            &area,
            y,
            active,
        );
        y = draw_divider(painter, &area, y);
        for (c, category) in CATEGORIES.iter().enumerate() {
            let Ok(c) = u8::try_from(c) else {
                break;
            };
            let view = View::Category(c);
            let badge = NavBadge {
                count: searching.then(|| self.found_count_in(settings, view)),
                warn: false,
            };
            y = self.draw_entry(painter, env, view, category.title, badge, &area, y, active);
        }
        y = draw_divider(painter, &area, y);
        let all = NavBadge {
            count: Some(if searching {
                self.found_total()
            } else {
                settings.catalog().len()
            }),
            warn: false,
        };
        y = self.draw_entry(
            painter,
            env,
            View::AllMods,
            "All mods",
            all,
            &area,
            y,
            active,
        );
        let reveal = self.nav_reveal(Pane::Sidebar);
        self.end_body(painter, Pane::Sidebar, area, y, reveal);
    }

    /// One window-1 entry at `y`: the outline pill when it is `active`, its label, its badge.
    /// Returns the y of the next entry.
    #[allow(clippy::too_many_arguments)]
    fn draw_entry(
        &mut self,
        painter: &mut Painter<'_>,
        env: &mut MenuEnv<'_>,
        view: View,
        label: &str,
        badge: NavBadge,
        area: &Area,
        y: f32,
        active: View,
    ) -> f32 {
        let s = area.s;
        let row = Rect::new(area.x, y, area.w, ENTRY_H * s);
        let hit = area.hit(painter, row);
        let hover =
            self.drag.is_none() && hit.is_some_and(|r| r.contains(self.cursor.0, self.cursor.1));
        let selected = view == active;
        let sidebar = Pane::Sidebar.index();
        let index = self.nav[sidebar].len();
        let focused =
            self.keyboard && self.pane == Pane::Sidebar && self.cursors[sidebar] == Some(index);
        let radius = 11.0 * s;
        if selected {
            painter.round_rect_shadow(row, radius, [10.0 * s; 4], white(0.06));
            painter.round_rect(
                row,
                radius,
                Fill::css_angle(row, 90.0, white(0.26), black(0.12)),
            );
            painter.round_rect_stroke(row, radius, 1.0, 0.5, white(0.54));
            painter.round_rect(
                Rect::new(row.x + 1.5 * s, row.y + 8.0 * s, 3.0 * s, row.h - 16.0 * s),
                1.5 * s,
                Fill::Solid(white(0.9)),
            );
        } else if hover {
            painter.round_rect(row, radius, Fill::Solid(white(0.09)));
            painter.round_rect_stroke(row, radius, 1.0, 0.5, white(0.13));
        }
        if focused {
            draw_focus_ring(painter, row, radius, s);
        }
        // While searching, an entry without results reads dimmer.
        let empty = self.searching() && badge.count == Some(0);
        let style = TextStyle::new(13.0 * s).weight(if selected {
            Weight::Semibold
        } else {
            Weight::Regular
        });
        let (_, lh) = env.text.metrics(style);
        let color = if empty {
            white(0.40)
        } else if selected || hover {
            white(0.98)
        } else {
            white(0.84)
        };
        shadowed(
            painter,
            env,
            label,
            row.x + 14.0 * s,
            row.y + (row.h - lh) * 0.5,
            style,
            color,
            s,
        );
        let mut right = row.right() - 12.0 * s;
        if let Some(count) = badge.count {
            let text = count.to_string();
            let style = TextStyle::new(11.0 * s);
            let tw = env.text.measure(&text, style, env.mc_font);
            let (_, blh) = env.text.metrics(style);
            shadowed(
                painter,
                env,
                &text,
                right - tw,
                row.y + (row.h - blh) * 0.5,
                style,
                white(if empty { 0.30 } else { 0.56 }),
                s,
            );
            right -= tw + 7.0 * s;
        }
        if badge.warn {
            // An amber disc with a "!" (two keys clash).
            let (cx, cy) = (right - 5.0 * s, row.y + row.h * 0.5);
            painter.circle(cx, cy, 5.0 * s, css(255, 196, 82, 0.95));
            painter.rect(
                Rect::new(cx - 0.65 * s, cy - 3.0 * s, 1.3 * s, 3.6 * s),
                black(0.85),
            );
            painter.rect(
                Rect::new(cx - 0.65 * s, cy + 1.5 * s, 1.3 * s, 1.3 * s),
                black(0.85),
            );
        }
        if let Some(r) = hit {
            self.hits.push((r, Target::Nav(view)));
        }
        self.nav[sidebar].push(NavRow {
            enter: Target::Nav(view),
            space: None,
            span: area.span(row.y, row.bottom()),
        });
        row.bottom() + ENTRY_GAP * s
    }

    /// The search field (moved from the seven-window menu's taskbar): the magnifier, the query
    /// with its caret or selection, the clear button.
    #[allow(clippy::too_many_lines)]
    fn draw_search_field(
        &mut self,
        painter: &mut Painter<'_>,
        env: &mut MenuEnv<'_>,
        field: Rect,
        area: &Area,
    ) {
        let s = area.s;
        let hit = area.hit(painter, field);
        let hovered = hit.is_some_and(|r| r.contains(self.cursor.0, self.cursor.1));
        let active = !self.search.is_empty();
        let typing = self.typing_to_search();
        let radius = 12.0 * s;
        painter.round_rect(
            field,
            radius,
            Fill::Solid(black(if hovered || active { 0.30 } else { 0.22 })),
        );
        painter.round_rect_stroke(
            field,
            radius,
            1.0,
            0.5,
            white(if active {
                0.58
            } else if hovered {
                0.40
            } else {
                0.24
            }),
        );
        let sidebar = Pane::Sidebar.index();
        if self.keyboard && self.pane == Pane::Sidebar && self.cursors[sidebar].is_none() {
            draw_focus_ring(painter, field, radius, s);
        }
        let label_style = TextStyle::new(12.5 * s).weight(Weight::Semibold);
        let (_, lh) = env.text.metrics(label_style);
        let cy = field.y + field.h * 0.5;
        // The magnifier: a ring and a handle.
        let (gx, gy) = (field.x + 16.0 * s, cy - 1.0 * s);
        painter.round_rect_stroke(
            Rect::new(gx - 5.0 * s, gy - 5.0 * s, 10.0 * s, 10.0 * s),
            5.0 * s,
            1.4 * s,
            0.6 * s,
            white(0.85),
        );
        painter.line(
            gx + 3.6 * s,
            gy + 3.6 * s,
            gx + 7.0 * s,
            gy + 7.0 * s,
            1.6 * s,
            white(0.85),
        );
        let text_x = field.x + 30.0 * s;
        let mut text_right = field.right() - 10.0 * s;
        if let Some(r) = hit {
            self.hits.push((r, Target::SearchField));
        }
        if active {
            // The clear button (×).
            let clear = Rect::new(field.right() - 28.0 * s, field.y, 24.0 * s, field.h);
            let (ccx, ccy) = clear.center();
            let d = 3.5 * s;
            let clear_hit = area.hit(painter, clear);
            let a = if clear_hit.is_some_and(|r| r.contains(self.cursor.0, self.cursor.1)) {
                0.95
            } else {
                0.6
            };
            painter.line(ccx - d, ccy - d, ccx + d, ccy + d, 1.4 * s, white(a));
            painter.line(ccx - d, ccy + d, ccx + d, ccy - d, 1.4 * s, white(a));
            text_right = clear.x - 2.0 * s;
            if let Some(r) = clear_hit {
                self.hits.push((r, Target::SearchClear));
            }
        }
        let room = (text_right - text_x).max(1.0);
        let saved = painter.clip();
        let clip = Rect::new(text_x - 1.0 * s, field.y, room + 2.0 * s, field.h);
        painter.set_clip(Some(saved.map_or(clip, |c| c.intersect(&clip))));
        if active {
            // Long searches show their end, where the caret is.
            let tw = env.text.measure(&self.search, label_style, env.mc_font);
            let tx = if tw > room {
                text_x + room - tw
            } else {
                text_x
            };
            if typing && self.search_selected {
                painter.rect(
                    Rect::new(tx, cy - lh * 0.5, tw, lh),
                    css(70, 130, 230, 0.65),
                );
            }
            shadowed(
                painter,
                env,
                &self.search,
                tx,
                cy - lh * 0.5,
                label_style,
                white(0.97),
                s,
            );
            let blink = (ms_since(self.opened_at, env.now) as u64 / 500).is_multiple_of(2);
            if typing && !self.search_selected && blink {
                painter.rect(
                    Rect::new(
                        (tx + tw + 1.0).min(text_x + room),
                        cy - lh * 0.5 + 1.0 * s,
                        1.0 * s.max(1.0),
                        lh - 2.0 * s,
                    ),
                    white(0.9),
                );
            }
        } else {
            let hint = if typing {
                "Type to search"
            } else {
                "Search mods & settings"
            };
            shadowed(
                painter,
                env,
                hint,
                text_x,
                cy - lh * 0.5,
                label_style.weight(Weight::Regular),
                white(0.5),
                s,
            );
        }
        painter.set_clip(saved);
    }
}
