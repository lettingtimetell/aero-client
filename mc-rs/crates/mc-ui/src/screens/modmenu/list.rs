//! Window 2, "What": the mods in what window 1 picked, under section
//! titles. Each row: a dot when the mod differs from a fresh install, its name (with a pin when it
//! is pinned to My setup), a summary line (its category in All mods and the search results) and
//! its switch. A click on the row shows the mod in window 3, the switch turns it on or off, a
//! right click pins or unpins it. While searching, window 2 lists the results: MODS, then
//! SETTINGS (one row per mod, with the settings that matched).

use mc_mods::menu_layout::{self, CATEGORIES};
use mc_mods::menu_search::{Hit, HitTarget, SearchResults};
use mc_mods::{DynamicRows, ModDescriptor, ModSettings};

use super::layout::{draw_title, Area};
use super::paint::{black, white, Fill, Painter, Rect};
use super::search::MenuSearch;
use super::state::{NavRow, Pane, Target, View};
use super::text::{TextStyle, Weight};
use super::widgets::{
    draw_focus_ring, draw_pin, draw_switch, draw_wrapped, fit_ellipsis, shadowed,
};
use super::{MenuEnv, ModMenuScreen};

pub(super) const SECTION_H: f32 = 24.0;
pub(super) const SECTION_GAP: f32 = 10.0;
pub(super) const ROW_H: f32 = 44.0;
pub(super) const ROW_GAP: f32 = 3.0;
pub(super) const ROW_RADIUS: f32 = 13.0;
/// [`MenuOrder::home`] of a mod the layout does not place.
pub(super) const NO_HOME: u8 = u8::MAX;
/// The longest summary line, in characters.
const SUMMARY_MAX: usize = 40;

/// Every catalog mod once, in menu order (window-1 order, then section order), with its home
/// category: the first category of [`CATEGORIES`] that lists it.
#[derive(Debug, Clone, Default)]
pub(super) struct MenuOrder {
    /// `(catalog index, home category)`, in menu order. A mod the layout does not place comes
    /// last with [`NO_HOME`] (the completeness test keeps that empty).
    pub(super) mods: Vec<(usize, u8)>,
    /// The home category per catalog index.
    pub(super) home: Vec<u8>,
    /// Catalog indices sorted by name (All mods).
    pub(super) by_name: Vec<usize>,
}

impl MenuOrder {
    pub(super) fn new(catalog: &[&'static ModDescriptor]) -> MenuOrder {
        let mut home = vec![NO_HOME; catalog.len()];
        let mut mods = Vec::with_capacity(catalog.len());
        for (c, category) in CATEGORIES.iter().enumerate() {
            let c = u8::try_from(c).unwrap_or(NO_HOME);
            for section in category.sections {
                for id in section.mods {
                    if let Some(m) = catalog.iter().position(|d| d.id == *id) {
                        if home[m] == NO_HOME {
                            home[m] = c;
                            mods.push((m, c));
                        }
                    }
                }
            }
        }
        for (m, &h) in home.iter().enumerate() {
            if h == NO_HOME {
                mods.push((m, NO_HOME));
            }
        }
        let mut by_name: Vec<usize> = (0..catalog.len()).collect();
        by_name.sort_by_cached_key(|&m| catalog[m].name.to_ascii_lowercase());
        MenuOrder {
            mods,
            home,
            by_name,
        }
    }

    /// The title of mod `m`'s home category ("" for a mod the layout does not place).
    pub(super) fn category_title(&self, m: usize) -> &'static str {
        self.home
            .get(m)
            .and_then(|&c| CATEGORIES.get(usize::from(c)))
            .map_or("", |category| category.title)
    }
}

/// This menu's search: The engine (`mc_mods::menu_search`: both spellings, optional
/// spaces, keywords, ranking) through its menu-side [`MenuSearch`], made at the first query.
#[derive(Debug, Default)]
pub(super) struct Found {
    search: Option<MenuSearch>,
}

impl Found {
    /// Bring the results up to date with `query` and this frame's dynamic rows (it searches only
    /// when one of them changed).
    pub(super) fn update(
        &mut self,
        catalog: &'static [&'static ModDescriptor],
        query: &str,
        rows: &[DynamicRows],
    ) {
        if self.search.is_none() && query.split_whitespace().next().is_none() {
            return;
        }
        self.search
            .get_or_insert_with(|| MenuSearch::new(catalog))
            .update(query, rows);
    }

    /// The current results (`None` before the first search).
    pub(super) fn results(&self) -> Option<&SearchResults> {
        self.search.as_ref().map(MenuSearch::results)
    }

    /// Whether setting `o` of mod `m` is a result (it is then never hidden in window 3).
    pub(super) fn option_hit(&self, m: usize, o: usize) -> bool {
        self.results()
            .is_some_and(|results| results.option_hit(m, o))
    }

    /// The settings of mod `m` that matched, best first.
    pub(super) fn options_of(&self, m: usize) -> Vec<usize> {
        self.results().map_or_else(Vec::new, |results| {
            results
                .settings_of(m)
                .filter_map(|hit| match hit.target {
                    HitTarget::Option(o) => Some(o),
                    HitTarget::Row(_) => None,
                })
                .collect()
        })
    }

    /// Every mod with a result, once each: the mods that matched first (best first), then the
    /// mods found through their settings.
    fn each_mod(&self) -> impl Iterator<Item = usize> + '_ {
        let results = self.results();
        let mods = results
            .into_iter()
            .flat_map(|results| results.mods.iter().map(|hit| hit.mod_index));
        let through_settings = results.into_iter().flat_map(|results| {
            results
                .setting_mods()
                .filter(move |&m| !results.is_mod_hit(m))
        });
        mods.chain(through_settings)
    }
}

/// Whether mod `m` differs from a fresh install: its switch, or any of its settings (not where
/// its HUD elements sit).
pub(super) fn mod_changed(settings: &ModSettings, m: usize) -> bool {
    settings.catalog().get(m).is_some_and(|descriptor| {
        settings.enabled_at(m) != descriptor.default_enabled || !settings.options_are_default_at(m)
    })
}

/// The line under a mod's name: its summary from `menu_layout::MOD_META`, else the start of
/// its description cut at a word (at most [`SUMMARY_MAX`] characters).
pub(super) fn summary(descriptor: &ModDescriptor) -> &'static str {
    if let Some(meta) = menu_layout::meta(descriptor.id) {
        if !meta.summary.is_empty() {
            return meta.summary;
        }
    }
    let text = descriptor.description.trim();
    let first = text
        .split(". ")
        .next()
        .unwrap_or(text)
        .trim_end_matches('.');
    if first.chars().count() <= SUMMARY_MAX {
        return first;
    }
    let mut end = 0;
    for (i, c) in first.char_indices() {
        if first[..i].chars().count() >= SUMMARY_MAX {
            break;
        }
        if c == ' ' {
            end = i;
        }
    }
    first[..end].trim_end_matches([',', ' '])
}

/// A section title: 10 px, bold, spaced capitals, 72% white (the design's `.section-title`).
fn draw_section_title(
    painter: &mut Painter<'_>,
    env: &mut MenuEnv<'_>,
    area: &Area,
    y: f32,
    title: &str,
) -> f32 {
    let s = area.s;
    shadowed(
        painter,
        env,
        title,
        area.x + 11.0 * s,
        y + 5.0 * s,
        TextStyle::new(10.0 * s).weight(Weight::Bold).caps(0.16),
        white(0.72),
        s,
    );
    y + SECTION_H * s
}

impl ModMenuScreen {
    /// Refresh the changed-from-a-fresh-install flags when the settings generation moved.
    pub(super) fn refresh_changed(&mut self, settings: &ModSettings) {
        let generation = settings.generation();
        let count = settings.catalog().len();
        if self.changed_gen == generation && self.changed.len() == count {
            return;
        }
        self.changed_gen = generation;
        self.changed.clear();
        self.changed
            .extend((0..count).map(|m| mod_changed(settings, m)));
    }

    /// Whether mod `m` is in My setup: changed from a fresh install, or pinned.
    pub(super) fn in_my_setup(&self, settings: &ModSettings, m: usize) -> bool {
        self.is_changed(m) || settings.pinned_at(m)
    }

    /// How many mods My setup lists.
    pub(super) fn my_setup_len(&self, settings: &ModSettings) -> usize {
        (0..settings.catalog().len())
            .filter(|&m| self.in_my_setup(settings, m))
            .count()
    }

    /// Whether window 2 lists mod `m` in `view` (search narrowing and window-1 counts).
    pub(super) fn view_lists(&self, settings: &ModSettings, view: View, m: usize) -> bool {
        match view {
            View::MySetup => self.in_my_setup(settings, m),
            View::Category(c) => CATEGORIES
                .get(usize::from(c))
                .is_some_and(|category| category.lists(settings.catalog()[m].id)),
            View::AllMods => true,
            View::Keybinds | View::Profiles => false,
        }
    }

    /// The first mod `view` lists (what window 3 shows when nothing is selected yet).
    pub(super) fn first_in_view(&self, settings: &ModSettings) -> Option<usize> {
        let catalog = settings.catalog();
        match self.view.unwrap_or(View::MySetup) {
            View::MySetup => self
                .order
                .mods
                .iter()
                .map(|&(m, _)| m)
                .find(|&m| self.in_my_setup(settings, m)),
            View::Category(c) => CATEGORIES.get(usize::from(c)).and_then(|category| {
                category
                    .sections
                    .iter()
                    .flat_map(|section| section.mods.iter())
                    .find_map(|id| catalog.iter().position(|d| d.id == *id))
            }),
            View::AllMods => self.order.by_name.first().copied(),
            View::Keybinds | View::Profiles => None,
        }
    }

    /// How many mods with a search result `view` lists (window 1's counts while searching).
    pub(super) fn found_count_in(&self, settings: &ModSettings, view: View) -> usize {
        self.found
            .each_mod()
            .filter(|&m| self.view_lists(settings, view, m))
            .count()
    }

    /// How many mods have a search result.
    pub(super) fn found_total(&self) -> usize {
        self.found.results().map_or(0, SearchResults::mod_count)
    }

    /// Bring this frame's search results up to date (the engine re-runs only when the
    /// query or the dynamic rows changed).
    pub(super) fn refresh_found(&mut self, env: &MenuEnv<'_>) {
        let query = if self.searching() {
            self.search.as_str()
        } else {
            ""
        };
        self.found.update(env.settings.catalog(), query, env.rows);
    }

    /// The search's best result as a window-2 target: what Enter opens and the list marks.
    pub(super) fn best_target(&self) -> Option<Target> {
        let best = self.found.results().and_then(SearchResults::best)?;
        Some(match best {
            Hit::Mod(hit) => Target::ModRow(hit.mod_index),
            Hit::Setting(hit) => Target::Found(hit.mod_index),
        })
    }

    /// The span of `pane`'s keyboard row when the arrow keys just moved it, to scroll it into
    /// view (once).
    pub(super) fn nav_reveal(&mut self, pane: Pane) -> Option<(f32, f32)> {
        if self.reveal_nav != Some(pane) {
            return None;
        }
        self.reveal_nav = None;
        let i = pane.index();
        self.cursors[i]
            .and_then(|cursor| self.nav[i].get(cursor))
            .map(|row| row.span)
    }

    /// Window 2 at `rect`: the search results while searching, else the view's list.
    pub(super) fn draw_list(
        &mut self,
        painter: &mut Painter<'_>,
        env: &mut MenuEnv<'_>,
        rect: Rect,
        s: f32,
    ) {
        if self.searching() {
            self.draw_results(painter, env, rect, s);
        } else {
            match self.view.unwrap_or(View::MySetup) {
                // The views draw window 2 themselves.
                View::Keybinds => self.draw_keybinds_list(painter, env, rect, s),
                View::Profiles => self.draw_profiles_list(painter, env, rect, s),
                View::MySetup => self.draw_my_setup(painter, env, rect, s),
                View::Category(c) => self.draw_category(painter, env, rect, s, c),
                View::AllMods => self.draw_all_mods(painter, env, rect, s),
            }
        }
        self.reveal = None;
    }

    fn draw_my_setup(
        &mut self,
        painter: &mut Painter<'_>,
        env: &mut MenuEnv<'_>,
        rect: Rect,
        s: f32,
    ) {
        let settings = env.settings;
        let count = settings.catalog().len();
        let changed = (0..count).filter(|&m| self.is_changed(m)).count();
        let pinned = (0..count).filter(|&m| settings.pinned_at(m)).count();
        let subtitle = match (changed, pinned) {
            (0, 0) => "Nothing changed yet".to_owned(),
            (c, 0) => format!("{c} changed from a fresh install"),
            (0, p) => format!("{p} pinned"),
            (c, p) => format!("{c} changed  \u{b7}  {p} pinned"),
        };
        draw_title(painter, env, rect, "My setup", &subtitle, 0.0, s);
        let area = self.begin_body(painter, rect, Pane::List, s);
        let blurb = if changed + pinned == 0 {
            "Mods you turn on or change show up here, so you can find them again. To keep any \
             mod here, pin it: right-click it in a list, or use Pin in its settings."
        } else {
            "What you've changed from a fresh install, and the mods you've pinned. Right-click \
             a mod to pin or unpin it."
        };
        let mut y = draw_wrapped(
            painter,
            env,
            blurb,
            area.x + 4.0 * s,
            area.y,
            area.w - 8.0 * s,
            TextStyle::new(11.5 * s),
            white(0.62),
            s,
        ) + 8.0 * s;
        let mut reveal = None;
        let mut first = true;
        for c in 0..=CATEGORIES.len() {
            let home = u8::try_from(c)
                .ok()
                .filter(|_| c < CATEGORIES.len())
                .unwrap_or(NO_HOME);
            let rows: Vec<usize> = self
                .order
                .mods
                .iter()
                .filter(|&&(m, h)| h == home && self.in_my_setup(settings, m))
                .map(|&(m, _)| m)
                .collect();
            if rows.is_empty() {
                continue;
            }
            if !first {
                y += SECTION_GAP * s;
            }
            first = false;
            let title = CATEGORIES.get(c).map_or("Other", |category| category.title);
            y = draw_section_title(painter, env, &area, y, title);
            for m in rows {
                let line = summary(settings.catalog()[m]);
                y = self.draw_mod_row(painter, env, m, &area, y, line, &mut reveal);
            }
        }
        let reveal = self.nav_reveal(Pane::List).or(reveal);
        self.end_body(painter, Pane::List, area, y, reveal);
    }

    fn draw_category(
        &mut self,
        painter: &mut Painter<'_>,
        env: &mut MenuEnv<'_>,
        rect: Rect,
        s: f32,
        c: u8,
    ) {
        let Some(category) = CATEGORIES.get(usize::from(c)) else {
            self.draw_my_setup(painter, env, rect, s);
            return;
        };
        let settings = env.settings;
        let catalog = settings.catalog();
        let count: usize = category
            .sections
            .iter()
            .map(|section| section.mods.len())
            .sum();
        let subtitle = if count == 1 {
            "1 mod".to_owned()
        } else {
            format!("{count} mods")
        };
        draw_title(painter, env, rect, category.title, &subtitle, 0.0, s);
        let area = self.begin_body(painter, rect, Pane::List, s);
        let mut y = draw_wrapped(
            painter,
            env,
            category.description,
            area.x + 4.0 * s,
            area.y,
            area.w - 8.0 * s,
            TextStyle::new(11.5 * s),
            white(0.64),
            s,
        ) + 6.0 * s;
        let mut reveal = None;
        let mut first = true;
        for section in category.sections {
            let rows: Vec<usize> = section
                .mods
                .iter()
                .filter_map(|id| catalog.iter().position(|d| d.id == *id))
                .collect();
            if rows.is_empty() {
                continue;
            }
            if !first {
                y += SECTION_GAP * s;
            }
            first = false;
            y = draw_section_title(painter, env, &area, y, section.title);
            for m in rows {
                y = self.draw_mod_row(painter, env, m, &area, y, summary(catalog[m]), &mut reveal);
            }
        }
        let reveal = self.nav_reveal(Pane::List).or(reveal);
        self.end_body(painter, Pane::List, area, y, reveal);
    }

    fn draw_all_mods(
        &mut self,
        painter: &mut Painter<'_>,
        env: &mut MenuEnv<'_>,
        rect: Rect,
        s: f32,
    ) {
        let count = env.settings.catalog().len();
        let subtitle = format!("{count} mods, A to Z");
        draw_title(painter, env, rect, "All mods", &subtitle, 0.0, s);
        let area = self.begin_body(painter, rect, Pane::List, s);
        let mut y = area.y;
        let mut reveal = None;
        let mut next = 0;
        // By position: drawing a row needs `&mut self`, so the order is not borrowed across it.
        while let Some(&m) = self.order.by_name.get(next) {
            next += 1;
            let line = self.order.category_title(m);
            y = self.draw_mod_row(painter, env, m, &area, y, line, &mut reveal);
        }
        let reveal = self.nav_reveal(Pane::List).or(reveal);
        self.end_body(painter, Pane::List, area, y, reveal);
    }

    /// The search results: MODS (ranked, each with its category), then SETTINGS (one row per
    /// mod), narrowed to the window-1 entry clicked while searching.
    #[allow(clippy::too_many_lines)]
    fn draw_results(
        &mut self,
        painter: &mut Painter<'_>,
        env: &mut MenuEnv<'_>,
        rect: Rect,
        s: f32,
    ) {
        let settings = env.settings;
        let narrow = self.narrow.filter(|view| *view != View::AllMods);
        let shows = |screen: &ModMenuScreen, m: usize| {
            narrow.is_none_or(|view| screen.view_lists(settings, view, m))
        };
        let (mods, groups): (Vec<usize>, Vec<usize>) = match self.found.results() {
            Some(results) => (
                results
                    .mods
                    .iter()
                    .map(|hit| hit.mod_index)
                    .filter(|&m| shows(self, m))
                    .collect(),
                results.setting_mods().filter(|&m| shows(self, m)).collect(),
            ),
            None => (Vec::new(), Vec::new()),
        };
        let total = mods.len() + groups.iter().filter(|m| !mods.contains(m)).count();
        self.search_found = total;
        // The best result is marked: Enter opens it (while no row has the keyboard).
        self.best = self.best_target();
        let query = self.search.trim().to_owned();
        let subtitle = if total == 0 {
            format!("No matches for \u{201c}{query}\u{201d}")
        } else {
            format!("\u{201c}{query}\u{201d}: {total} found")
        };
        draw_title(painter, env, rect, "Results", &subtitle, 0.0, s);
        let area = self.begin_body(painter, rect, Pane::List, s);
        let mut y = area.y;
        let note_style = TextStyle::new(11.5 * s);
        if let Some(view) = narrow {
            let name = match view {
                View::Category(c) => CATEGORIES.get(usize::from(c)).map_or("", |c| c.title),
                _ => "My setup",
            };
            let text = format!("Only results in {name}. All mods shows every result.");
            y = draw_wrapped(
                painter,
                env,
                &text,
                area.x + 4.0 * s,
                y,
                area.w - 8.0 * s,
                note_style,
                white(0.62),
                s,
            ) + 6.0 * s;
        }
        if total == 0 {
            y = draw_wrapped(
                painter,
                env,
                "Nothing matches. Try fewer words, or another word for it. Esc clears the \
                 search.",
                area.x + 4.0 * s,
                y,
                area.w - 8.0 * s,
                note_style,
                white(0.62),
                s,
            );
        }
        let mut reveal = None;
        if !mods.is_empty() {
            y = draw_section_title(painter, env, &area, y, "Mods");
            for m in mods {
                let line = self.order.category_title(m);
                y = self.draw_mod_row(painter, env, m, &area, y, line, &mut reveal);
            }
        }
        if !groups.is_empty() {
            if y > area.y {
                y += SECTION_GAP * s;
            }
            y = draw_section_title(painter, env, &area, y, "Settings");
            for m in groups {
                y = self.draw_found_row(painter, env, m, &area, y);
            }
        }
        let reveal = self.nav_reveal(Pane::List).or(reveal);
        self.end_body(painter, Pane::List, area, y, reveal);
    }

    /// The keyboard's focus ring on a row of window 2, or, while no row has the keyboard, a
    /// softer ring on the search's best result (what Enter opens).
    fn draw_row_marks(
        &self,
        painter: &mut Painter<'_>,
        row: Rect,
        radius: f32,
        s: f32,
        focused: bool,
        target: Target,
    ) {
        if focused {
            draw_focus_ring(painter, row, radius, s);
        } else if self.best == Some(target) && self.cursors[Pane::List.index()].is_none() {
            painter.round_rect_stroke(row.inset(-s), radius + s, 1.2 * s, 0.5, white(0.58));
        }
    }

    /// A mod's row in window 2: the changed dot, the name (and pin), `second` under it, the
    /// switch. Returns the y below it.
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    fn draw_mod_row(
        &mut self,
        painter: &mut Painter<'_>,
        env: &mut MenuEnv<'_>,
        m: usize,
        area: &Area,
        y: f32,
        second: &str,
        reveal: &mut Option<(f32, f32)>,
    ) -> f32 {
        let s = area.s;
        let settings = env.settings;
        let descriptor = settings.catalog()[m];
        let enabled = settings.enabled_at(m);
        let selected = self.selected == Some(m);
        let row = Rect::new(area.x, y, area.w, ROW_H * s);
        let hit = area.hit(painter, row);
        let hover =
            self.drag.is_none() && hit.is_some_and(|r| r.contains(self.cursor.0, self.cursor.1));
        let list = Pane::List.index();
        let index = self.nav[list].len();
        let focused = self.keyboard && self.pane == Pane::List && self.cursors[list] == Some(index);
        let radius = ROW_RADIUS * s;
        if selected {
            // .nav-item.active: the white outline pill.
            painter.round_rect_shadow(row, radius, [10.0 * s; 4], white(0.06));
            painter.round_rect(
                row,
                radius,
                Fill::css_angle(row, 90.0, white(0.26), black(0.12)),
            );
            painter.round_rect_stroke(row, radius, 1.0, 0.5, white(0.54));
            // inset 3px 0 0 rgba(255,255,255,.9): the bright left edge.
            painter.round_rect(
                Rect::new(row.x + 1.5 * s, row.y + 11.0 * s, 3.0 * s, row.h - 22.0 * s),
                1.5 * s,
                Fill::Solid(white(0.9)),
            );
        } else if hover {
            // .nav-item:hover
            painter.round_rect(row, radius, Fill::Solid(white(0.09)));
            painter.round_rect_stroke(row, radius, 1.0, 0.5, white(0.13));
        }
        self.draw_row_marks(painter, row, radius, s, focused, Target::ModRow(m));
        // The dot: changed from a fresh install (it no longer repeats the switch).
        if self.is_changed(m) {
            let (dx, dy) = (row.x + 13.0 * s, row.y + 15.0 * s);
            painter.circle(dx, dy, 6.0 * s, white(0.10));
            painter.circle(dx, dy, 3.0 * s, white(0.95));
        }
        let sw = Rect::new(
            row.right() - 46.0 * s,
            row.y + (row.h - 18.0 * s) * 0.5,
            34.0 * s,
            18.0 * s,
        );
        let name_x = row.x + 25.0 * s;
        let room = (sw.x - 10.0 * s - name_x).max(1.0);
        let pinned = settings.pinned_at(m);
        let pin_w = if pinned { 16.0 * s } else { 0.0 };
        let mut name_style = TextStyle::new(13.5 * s).weight(if selected {
            Weight::Semibold
        } else {
            Weight::Regular
        });
        let natural = env.text.measure(descriptor.name, name_style, env.mc_font);
        if natural + pin_w > room {
            name_style.size = (name_style.size * (room - pin_w).max(1.0) / natural).max(11.0 * s);
        }
        let saved = painter.clip();
        let text_clip = Rect::new(name_x - 2.0 * s, row.y, room + 2.0 * s, row.h);
        painter.set_clip(Some(saved.map_or(text_clip, |c| c.intersect(&text_clip))));
        let name_color = if enabled || selected || hover {
            white(0.97)
        } else {
            white(0.80)
        };
        let advance = shadowed(
            painter,
            env,
            descriptor.name,
            name_x,
            row.y + 6.0 * s,
            name_style,
            name_color,
            s,
        );
        if pinned {
            let (_, lh) = env.text.metrics(name_style);
            draw_pin(
                painter,
                name_x + advance + 9.0 * s,
                row.y + 6.0 * s + lh * 0.5,
                s,
                white(0.86),
            );
        }
        let second_style = TextStyle::new(11.0 * s);
        let second = fit_ellipsis(env, second, room, second_style);
        shadowed(
            painter,
            env,
            &second,
            name_x,
            row.y + 25.0 * s,
            second_style,
            white(if enabled || selected { 0.68 } else { 0.58 }),
            s,
        );
        painter.set_clip(saved);
        let t = self.switch_anim.get(m).copied().unwrap_or(0.0);
        draw_switch(painter, sw, t, hover);
        if let Some(r) = hit {
            self.hits.push((r, Target::ModRow(m)));
            if let Some(sw_hit) = area.hit(painter, sw.inset(-4.0 * s)) {
                self.hits.push((sw_hit.intersect(&r), Target::Switch(m)));
            }
        }
        let span = area.span(row.y, row.bottom());
        self.nav[list].push(NavRow {
            enter: Target::ModRow(m),
            space: Some(Target::Switch(m)),
            span,
        });
        if self.reveal == Some(m) {
            *reveal = Some(span);
        }
        row.bottom() + ROW_GAP * s
    }

    /// A search result's settings under one mod: the mod's name, then the matching settings'
    /// labels. A click opens the mod in window 3 at those settings.
    fn draw_found_row(
        &mut self,
        painter: &mut Painter<'_>,
        env: &mut MenuEnv<'_>,
        m: usize,
        area: &Area,
        y: f32,
    ) -> f32 {
        let s = area.s;
        let settings = env.settings;
        let descriptor = settings.catalog()[m];
        // The settings (and dynamic rows) that matched, best first.
        let mut labels = String::new();
        if let Some(results) = self.found.results() {
            let rows = env.rows.get(m).map_or(&[][..], DynamicRows::rows);
            for hit in results.settings_of(m) {
                let label = match hit.target {
                    HitTarget::Option(o) => descriptor.options.get(o).map(|option| option.label),
                    HitTarget::Row(key) => rows
                        .iter()
                        .find(|row| row.key == key)
                        .map(|row| row.label.as_str()),
                };
                if let Some(label) = label {
                    if !labels.is_empty() {
                        labels.push_str(", ");
                    }
                    labels.push_str(label);
                }
            }
        }
        let row = Rect::new(area.x, y, area.w, ROW_H * s);
        let hit = area.hit(painter, row);
        let hover =
            self.drag.is_none() && hit.is_some_and(|r| r.contains(self.cursor.0, self.cursor.1));
        let list = Pane::List.index();
        let index = self.nav[list].len();
        let focused = self.keyboard && self.pane == Pane::List && self.cursors[list] == Some(index);
        let selected = self.selected == Some(m) && !self.flash_options.is_empty();
        let radius = ROW_RADIUS * s;
        if selected {
            painter.round_rect(row, radius, Fill::Solid(white(0.16)));
            painter.round_rect_stroke(row, radius, 1.0, 0.5, white(0.42));
        } else if hover {
            painter.round_rect(row, radius, Fill::Solid(white(0.09)));
            painter.round_rect_stroke(row, radius, 1.0, 0.5, white(0.13));
        }
        self.draw_row_marks(painter, row, radius, s, focused, Target::Found(m));
        let x = row.x + 14.0 * s;
        let room = (row.right() - 30.0 * s - x).max(1.0);
        let saved = painter.clip();
        let text_clip = Rect::new(x - 2.0 * s, row.y, room + 2.0 * s, row.h);
        painter.set_clip(Some(saved.map_or(text_clip, |c| c.intersect(&text_clip))));
        shadowed(
            painter,
            env,
            descriptor.name,
            x,
            row.y + 6.0 * s,
            TextStyle::new(13.5 * s),
            white(0.97),
            s,
        );
        let labels_style = TextStyle::new(11.0 * s);
        let line = fit_ellipsis(env, &labels, room, labels_style);
        shadowed(
            painter,
            env,
            &line,
            x,
            row.y + 25.0 * s,
            labels_style,
            white(0.68),
            s,
        );
        painter.set_clip(saved);
        // A chevron: the row opens the mod.
        let (cx, cy) = (row.right() - 16.0 * s, row.y + row.h * 0.5);
        let a = if hover { 0.9 } else { 0.5 };
        painter.line(
            cx - 1.5 * s,
            cy - 3.5 * s,
            cx + 2.0 * s,
            cy,
            1.3 * s,
            white(a),
        );
        painter.line(
            cx + 2.0 * s,
            cy,
            cx - 1.5 * s,
            cy + 3.5 * s,
            1.3 * s,
            white(a),
        );
        if let Some(r) = hit {
            self.hits.push((r, Target::Found(m)));
        }
        self.nav[list].push(NavRow {
            enter: Target::Found(m),
            space: None,
            span: area.span(row.y, row.bottom()),
        });
        row.bottom() + ROW_GAP * s
    }
}
