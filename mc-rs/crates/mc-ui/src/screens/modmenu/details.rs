//! Window 3, "How": the selected mod's title bar (its name, where it
//! lives, its switch), a header (the full description, notes, "Move on screen", "Reset" and
//! "Pin") and its settings at full width: under headings, with a dot on every changed setting (a
//! reset arrow on hover; a right click on the label resets it too), dimmed or hidden while they
//! do nothing, tabbed pickers and paired rows.
//!
//! The option widgets, the colour editor, colour grading's profiles and the lanes' dynamic rows
//! come from the earlier seven-window menu's inline panel.
//!
//! The blocks, headings, paired rows, tabs and the hide/dim rules come from the `mc_mods::menu_rules` (`settings_layout`, `option_state`).

use mc_mods::menu_layout;
use mc_mods::menu_rules::{self, Heading, LayoutRow, OptionRule, OptionState};
use mc_mods::{
    DynamicRows, ModDescriptor, ModSettings, OptionDescriptor, OptionKind, OptionValue, RowButton,
    RowIcon,
};

use super::keybinds::key_name;
use super::layout::{draw_title, Area, TITLE_H};
use super::paint::{self, black, css, white, Fill, Painter, Rect};
use super::state::{NavRow, Pane, Target, View};
use super::text::{TextStyle, Weight};
use super::widgets::{
    argb_to_hsva, draw_badge, draw_checker, draw_focus_ring, draw_icon, draw_pin,
    draw_slider_track, draw_switch, draw_wrapped, hsva_to_argb, ms_since, shadowed, still_armed,
};
use super::{MenuEnv, MenuInput, ModMenuScreen};

pub(super) const OPT_H: f32 = 32.0;
pub(super) const SLIDER_H: f32 = 46.0;
pub(super) const COLOR_EDITOR_H: f32 = 118.0;
pub(super) const DYN_H: f32 = 30.0;
/// A heading over a block of settings.
pub(super) const HEADING_H: f32 = 30.0;
/// The column left of each setting that holds its changed dot and reset arrow.
pub(super) const GUTTER: f32 = 14.0;
/// How long the settings a search result opened glow.
pub(super) const FLASH_MS: f32 = 1600.0;

/// Colour presets in the colour editor (Minecraft's chat palette, plus white and black).
pub(super) const PRESETS: [u32; 10] = [
    0xFFFF_FFFF,
    0xFFAA_AAAA,
    0xFFFF_5555,
    0xFFFF_AA00,
    0xFFFF_FF55,
    0xFF55_FF55,
    0xFF55_FFFF,
    0xFF55_55FF,
    0xFFFF_55FF,
    0xFF00_0000,
];

/// Mods whose expanded options offer named profiles (items 12/43: colour grading). Profiles
/// live in the mod's settings entry ([`ModSettings::profiles_at`]), so adding a mod here is all
/// another mod needs.
pub const PROFILE_MODS: &[&str] = &["color_grading"];

/// The "Profiles" label row and the name field row.
pub(super) const PROFILE_HEAD_H: f32 = 26.0;

/// Why a setting shows dimmed: the rule that dims it (the hover text is
/// [`OptionRule::reason`]), `None` when it shows normally.
type Why = Option<&'static OptionRule>;

/// The key of a block's open or closed state: (mod, block index).
fn group_key(m: usize, block: usize) -> (usize, u8) {
    (m, u8::try_from(block).unwrap_or(u8::MAX))
}

/// The setting whose rules decide whether `row` shows (a pair's left, a tab row's picker).
fn row_lead(row: &LayoutRow) -> usize {
    match row {
        LayoutRow::Option(o) | LayoutRow::Pair(o, _) => *o,
        LayoutRow::Tabs { picker, .. } => *picker,
    }
}

/// How tall a setting's own line is (design px): sliders have their track under the label.
fn option_h(option: &OptionDescriptor) -> f32 {
    match option.kind {
        OptionKind::Slider { .. } | OptionKind::SliderLabeled { .. } => SLIDER_H,
        _ => OPT_H,
    }
}

/// "HUD › Counters": the mod's home category and section.
fn breadcrumb(id: &str) -> String {
    menu_layout::home_category(id)
        .and_then(|category| {
            category
                .section_of(id)
                .map(|section| format!("{} \u{203a} {}", category.title, section.title))
        })
        .unwrap_or_default()
}

/// The setting a keyboard row of window 3 stands for.
pub(super) fn nav_option(target: Target) -> Option<(usize, usize)> {
    match target {
        Target::Toggle(m, o)
        | Target::Choice(m, o)
        | Target::Key(m, o)
        | Target::Text(m, o)
        | Target::Button(m, o)
        | Target::Swatch(m, o)
        | Target::Label(m, o) => Some((m, o)),
        _ => None,
    }
}

impl ModMenuScreen {
    /// Save mod `m`'s current options under the typed name (an empty name gets "Profile N"),
    /// then clear the field.
    pub(super) fn save_profile(&mut self, m: usize, input: &mut MenuInput<'_>) {
        if input
            .settings
            .save_profile_at(m, &self.profile_name)
            .is_some()
        {
            self.profile_name.clear();
        }
        self.delete_armed = None;
    }

    pub(super) fn set_color_channel(
        &mut self,
        settings: &mut ModSettings,
        m: usize,
        o: usize,
        channel: u8,
        track: Rect,
        x: f32,
    ) {
        let t = ((x - track.x) / track.w.max(1.0)).clamp(0.0, 1.0);
        self.color_hsv[usize::from(channel.min(3))] = t;
        let argb = hsva_to_argb(self.color_hsv);
        settings.set_value_at(m, o, OptionValue::Color(argb));
    }

    /// Window 3 at `rect`: The views first (Keybinds and Profiles may draw it), else the
    /// selected mod.
    pub(super) fn draw_details(
        &mut self,
        painter: &mut Painter<'_>,
        env: &mut MenuEnv<'_>,
        rect: Rect,
        s: f32,
    ) {
        // The views may draw window 3 themselves.
        if !self.searching() {
            let drawn = match self.view {
                Some(View::Keybinds) => self.draw_keybinds_details(painter, env, rect, s),
                Some(View::Profiles) => self.draw_profiles_details(painter, env, rect, s),
                _ => false,
            };
            if drawn {
                return;
            }
        }
        let settings = env.settings;
        let catalog = settings.catalog();
        let Some(m) = self.selected.filter(|&m| m < catalog.len()) else {
            draw_title(painter, env, rect, "Settings", "Nothing picked", 0.0, s);
            let area = self.begin_body(painter, rect, Pane::Details, s);
            let y = draw_wrapped(
                painter,
                env,
                "Pick a mod in the list to see its settings here.",
                area.x,
                area.y,
                area.w,
                TextStyle::new(12.0 * s),
                white(0.6),
                s,
            );
            self.end_body(painter, Pane::Details, area, y, None);
            return;
        };
        let descriptor = catalog[m];
        let interactive = self.interactive();
        // The title bar: the name, where it lives, its switch.
        let sw = Rect::new(
            rect.right() - 56.0 * s,
            rect.y + (TITLE_H * s - 20.0 * s) * 0.5,
            38.0 * s,
            20.0 * s,
        );
        draw_title(
            painter,
            env,
            rect,
            descriptor.name,
            &breadcrumb(descriptor.id),
            sw.w + 14.0 * s,
            s,
        );
        let sw_hit = sw.inset(-5.0 * s);
        let sw_hover = interactive && sw_hit.contains(self.cursor.0, self.cursor.1);
        let t = self.switch_anim.get(m).copied().unwrap_or(0.0);
        draw_switch(painter, sw, t, sw_hover);
        let details = Pane::Details.index();
        let focused =
            self.keyboard && self.pane == Pane::Details && self.cursors[details] == Some(0);
        if focused {
            draw_focus_ring(painter, sw, sw.h * 0.5, s);
        }
        if interactive {
            self.hits.push((sw_hit, Target::Switch(m)));
        }
        self.nav[details].push(NavRow {
            enter: Target::Switch(m),
            space: Some(Target::Switch(m)),
            span: (0.0, 0.0),
        });

        let area = self.begin_body(painter, rect, Pane::Details, s);
        let mut reveal = None;
        let mut y = self.draw_header(painter, env, m, &area, area.y);
        // Live warnings and other notes.
        y = self.draw_mod_warnings(painter, env, m, &area, y);
        if PROFILE_MODS.contains(&descriptor.id) {
            y = self.draw_profiles(
                painter,
                env,
                m,
                (area.x, y, area.w),
                s,
                area.clip,
                area.interactive,
                false,
            ) + 6.0 * s;
        }
        y = self.draw_settings(painter, env, m, &area, y, &mut reveal);
        let dyn_rows = env.rows.get(m).map_or(&[][..], DynamicRows::rows);
        if !dyn_rows.is_empty() {
            y += 8.0 * s;
            for row in dyn_rows {
                y = self.draw_dynamic_row(
                    painter,
                    env,
                    m,
                    row,
                    area.x + GUTTER * s,
                    y,
                    area.w - GUTTER * s,
                    s,
                    area.clip,
                    area.interactive,
                );
            }
        }
        if reveal.is_none() && self.reveal_nav == Some(Pane::Details) {
            reveal = self.cursors[details]
                .and_then(|i| self.nav[details].get(i))
                .map(|row| row.span);
        }
        if self.reveal_nav == Some(Pane::Details) {
            self.reveal_nav = None;
        }
        self.reveal_option = None;
        self.end_body(painter, Pane::Details, area, y, reveal);
    }

    /// The header under the title bar: the full description, the notes, and the buttons
    /// ("Move on screen", "Reset settings", the pin). Returns the y below it.
    fn draw_header(
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
        let enabled = settings.enabled_at(m);
        let mut y = draw_wrapped(
            painter,
            env,
            descriptor.description,
            area.x,
            y,
            area.w,
            TextStyle::new(12.0 * s),
            white(0.80),
            s,
        );
        let note_style = TextStyle::new(11.0 * s);
        // "Part of vanilla Minecraft: on is the normal game." / "On by default in Aero."
        if let Some(note) = menu_layout::default_on_note(descriptor) {
            y += 4.0 * s;
            y = draw_wrapped(
                painter,
                env,
                note,
                area.x,
                y,
                area.w,
                note_style,
                white(0.52),
                s,
            );
        }
        // Settings that work with the mod off (`menu_rules::ALWAYS_APPLY`) must not be said to
        // wait for it.
        let (count, while_off) = descriptor
            .options
            .iter()
            .filter(|option| option.kind.is_visible())
            .fold((0, 0), |(count, while_off), option| {
                let always = menu_rules::applies_while_off(descriptor.id, option.id);
                (count + 1, while_off + usize::from(always))
            });
        let off_note = match (enabled, count, while_off) {
            (false, 1.., 0) => Some("Off: these settings apply when you turn it on."),
            (false, count, while_off) if while_off < count => {
                Some("Off: most of these settings apply only when you turn it on.")
            }
            _ => None,
        };
        if let Some(off_note) = off_note {
            y += 4.0 * s;
            y = draw_wrapped(
                painter,
                env,
                off_note,
                area.x,
                y,
                area.w,
                note_style,
                css(255, 222, 168, 0.86),
                s,
            );
        }
        y += 10.0 * s;
        let h = 26.0 * s;
        let details = Pane::Details.index();
        let mut x = area.x;
        // "Move on screen" (returns the x after its button).
        x = self.draw_move_on_screen(painter, env, m, area, (x, y));
        if descriptor.options.iter().any(|o| o.kind.is_persisted()) {
            let can = !settings.options_are_default_at(m);
            let armed = can
                && self
                    .reset_armed
                    .is_some_and(|(armed_m, at)| armed_m == m && still_armed(at, env.now));
            let label = if armed {
                "Click to confirm"
            } else {
                "Reset settings"
            };
            let r = self.draw_button(painter, env, label, (x, y), h, can, armed, None, area);
            if let Some(hit) = area.hit(painter, r) {
                if can {
                    self.hits.push((hit, Target::ResetMod(m)));
                    self.nav[details].push(NavRow {
                        enter: Target::ResetMod(m),
                        space: None,
                        span: area.span(r.y, r.bottom()),
                    });
                } else if hit.contains(self.cursor.0, self.cursor.1) {
                    self.set_tooltip("Every setting is at its default.", self.cursor);
                }
            }
            x = r.right() + 8.0 * s;
        }
        let pinned = settings.pinned_at(m);
        let label = if pinned { "Pinned" } else { "Pin to My setup" };
        let r = self.draw_button(
            painter,
            env,
            label,
            (x, y),
            h,
            true,
            false,
            Some(pinned),
            area,
        );
        if let Some(hit) = area.hit(painter, r) {
            self.hits.push((hit, Target::Pin(m)));
            if hit.contains(self.cursor.0, self.cursor.1) {
                let tip = if pinned {
                    "In My setup even with nothing changed. Click to unpin."
                } else {
                    "Keep this mod in My setup even with nothing changed. Right-click a mod in \
                     the list to pin it too."
                };
                self.set_tooltip(tip, self.cursor);
            }
        }
        self.nav[details].push(NavRow {
            enter: Target::Pin(m),
            space: Some(Target::Pin(m)),
            span: area.span(r.y, r.bottom()),
        });
        y + h + 12.0 * s
    }

    /// A text button with its left edge at `x`, `h` tall (greyed out when not `enabled`, red
    /// while `armed`; `pin` draws the pin glyph, lit when `Some(true)`). Returns its rectangle.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn draw_button(
        &self,
        painter: &mut Painter<'_>,
        env: &mut MenuEnv<'_>,
        label: &str,
        (x, y): (f32, f32),
        h: f32,
        enabled: bool,
        armed: bool,
        pin: Option<bool>,
        area: &Area,
    ) -> Rect {
        let s = area.s;
        let style = TextStyle::new(11.5 * s).weight(Weight::Semibold);
        let tw = env.text.measure(label, style, env.mc_font);
        let (_, lh) = env.text.metrics(style);
        let icon_w = if pin.is_some() { 15.0 * s } else { 0.0 };
        let r = Rect::new(x, y, tw + 22.0 * s + icon_w, h);
        let hover = enabled
            && area
                .hit(painter, r)
                .is_some_and(|hit| hit.contains(self.cursor.0, self.cursor.1));
        let lit = pin == Some(true);
        let fill = if armed {
            css(255, 110, 90, if hover { 0.42 } else { 0.30 })
        } else if !enabled {
            white(0.04)
        } else if lit {
            white(if hover { 0.30 } else { 0.22 })
        } else {
            white(if hover { 0.18 } else { 0.09 })
        };
        painter.round_rect(r, 8.0 * s, Fill::Solid(fill));
        let border = if armed {
            css(255, 150, 130, 0.8)
        } else if lit {
            white(0.62)
        } else {
            white(if enabled { 0.30 } else { 0.12 })
        };
        painter.round_rect_stroke(r, 8.0 * s, 1.0, 0.5, border);
        if let Some(on) = pin {
            draw_pin(
                painter,
                r.x + 13.0 * s,
                r.y + h * 0.5,
                s,
                white(if on { 1.0 } else { 0.78 }),
            );
        }
        shadowed(
            painter,
            env,
            label,
            r.x + 11.0 * s + icon_w,
            r.y + (h - lh) * 0.5,
            style,
            white(if enabled { 0.96 } else { 0.40 }),
            s,
        );
        r
    }

    /// The selected mod's settings, block by block (the `menu_rules::settings_layout`,
    /// worked out once per selected mod). Returns the y below them.
    fn draw_settings(
        &mut self,
        painter: &mut Painter<'_>,
        env: &mut MenuEnv<'_>,
        m: usize,
        area: &Area,
        y: f32,
        reveal: &mut Option<(f32, f32)>,
    ) -> f32 {
        let s = area.s;
        let settings = env.settings;
        let descriptor = settings.catalog()[m];
        let layout = match self.settings_layout.take() {
            Some((cached, layout)) if cached == m => layout,
            _ => menu_rules::settings_layout(descriptor),
        };
        let mut y = y;
        let mut shown: Vec<(usize, Why)> = Vec::new();
        for (b, block) in layout.blocks.iter().enumerate() {
            shown.clear();
            shown.extend(block.rows.iter().enumerate().filter_map(|(r, row)| {
                self.option_shown(settings, m, row_lead(row))
                    .map(|why| (r, why))
            }));
            if shown.is_empty() {
                continue;
            }
            if let Some(title) = block.heading.title() {
                let toggle = matches!(block.heading, Heading::Named(_)).then(|| group_key(m, b));
                // A setting a search result opened is never inside a closed group.
                let flashed = block
                    .rows
                    .iter()
                    .any(|row| row.options().any(|o| self.flash_options.contains(&o)));
                let collapsed = !flashed
                    && toggle
                        .is_some_and(|key| block.collapsed != self.toggled_groups.contains(&key));
                y = self.draw_heading(painter, env, area, y, title, toggle, collapsed, shown.len());
                if collapsed {
                    continue;
                }
                // A readout's "HUD style / Custom" choice goes
                // above STYLE's rows; folded on the HUD style, the rows are skipped.
                if block.heading == Heading::Style {
                    match self.draw_style_block(painter, env, m, area, y) {
                        super::extras::StyleBlock::Rows(top) => y = top,
                        super::extras::StyleBlock::Done(bottom) => {
                            y = bottom;
                            continue;
                        }
                    }
                }
            }
            for &(r, why) in &shown {
                y = match &block.rows[r] {
                    LayoutRow::Option(o) => {
                        self.draw_option_row(painter, env, m, *o, area, y, why, reveal)
                    }
                    LayoutRow::Pair(left, right) => {
                        self.draw_pair_row(painter, env, m, (*left, *right), area, y, why, reveal)
                    }
                    LayoutRow::Tabs { picker, edits } => {
                        self.draw_tabs(painter, env, m, *picker, edits, area, y, why, reveal)
                    }
                };
            }
            y += 4.0 * s;
        }
        self.settings_layout = Some((m, layout));
        y
    }

    /// How setting `o` of mod `m` shows right now (the `menu_rules::option_state`):
    /// `None` hidden, `Some(None)` normally, `Some(Some(rule))` dimmed. A setting found by search
    /// (now, or the result that opened it) is never hidden: it shows dimmed instead.
    fn option_shown(&self, settings: &ModSettings, m: usize, o: usize) -> Option<Why> {
        let mut state = menu_rules::option_state(settings, m, o);
        if self.flash_options.contains(&o) || self.found.option_hit(m, o) {
            state = state.for_search_hit();
        }
        match state {
            OptionState::Shown => Some(None),
            OptionState::Dimmed(rule) => Some(Some(rule)),
            OptionState::Hidden(_) => None,
        }
    }

    /// A heading in capitals with a hairline; a named group's heading has a chevron and folds.
    #[allow(clippy::too_many_arguments)]
    fn draw_heading(
        &mut self,
        painter: &mut Painter<'_>,
        env: &mut MenuEnv<'_>,
        area: &Area,
        y: f32,
        title: &str,
        toggle: Option<(usize, u8)>,
        collapsed: bool,
        count: usize,
    ) -> f32 {
        let s = area.s;
        let h = HEADING_H * s;
        let rect = Rect::new(area.x, y, area.w, h);
        let caps = TextStyle::new(10.0 * s).weight(Weight::Bold).caps(0.16);
        let (_, ch) = env.text.metrics(caps);
        let ty = y + h - ch - 6.0 * s;
        let mut x = area.x + 2.0 * s;
        if let Some((m, g)) = toggle {
            let hit = area.hit(painter, rect);
            let hover = hit.is_some_and(|r| r.contains(self.cursor.0, self.cursor.1));
            let (cx, cy) = (x + 4.0 * s, ty + ch * 0.5);
            let a = if hover { 0.95 } else { 0.6 };
            if collapsed {
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
            } else {
                painter.line(
                    cx - 3.5 * s,
                    cy - 1.5 * s,
                    cx,
                    cy + 2.0 * s,
                    1.3 * s,
                    white(a),
                );
                painter.line(
                    cx,
                    cy + 2.0 * s,
                    cx + 3.5 * s,
                    cy - 1.5 * s,
                    1.3 * s,
                    white(a),
                );
            }
            x += 14.0 * s;
            if let Some(r) = hit {
                self.hits.push((r, Target::Group(m, g)));
            }
            let details = Pane::Details.index();
            let index = self.nav[details].len();
            if self.keyboard && self.pane == Pane::Details && self.cursors[details] == Some(index) {
                draw_focus_ring(painter, rect, 8.0 * s, s);
            }
            self.nav[details].push(NavRow {
                enter: Target::Group(m, g),
                space: Some(Target::Group(m, g)),
                span: area.span(rect.y, rect.bottom()),
            });
        }
        let advance = shadowed(painter, env, title, x, ty, caps, white(0.72), s);
        let mut end = x + advance;
        if collapsed {
            let text = format!("{count}");
            end += 6.0 * s;
            end += shadowed(painter, env, &text, end, ty, caps, white(0.45), s);
        }
        let line_x = end + 10.0 * s;
        if line_x < area.right() {
            painter.rect(
                Rect::new(line_x, (ty + ch * 0.5).round(), area.right() - line_x, 1.0),
                white(0.10),
            );
        }
        y + h
    }

    /// The glow on settings a search result opened (fading over [`FLASH_MS`]).
    fn draw_flash(&self, painter: &mut Painter<'_>, env: &MenuEnv<'_>, o: usize, rect: Rect) {
        let Some(since) = self.flash_since else {
            return;
        };
        let t = ms_since(since, env.now);
        if t < FLASH_MS && self.flash_options.contains(&o) {
            let a = 0.30 * (1.0 - t / FLASH_MS);
            painter.round_rect(rect, rect.h.min(16.0), Fill::Solid(css(150, 200, 255, a)));
        }
    }

    /// The hover text of a setting's label: why it is dimmed, then what it does.
    fn option_tooltip(&mut self, descriptor: &ModDescriptor, option: &OptionDescriptor, why: Why) {
        let at = self.cursor;
        let description = option.description.trim();
        match why {
            Some(rule) => {
                let mut text = String::new();
                rule.reason(descriptor, &mut text);
                if !description.is_empty() {
                    text.push_str(". ");
                    text.push_str(description);
                }
                self.set_tooltip(&text, at);
            }
            None if !description.is_empty() => self.set_tooltip(description, at),
            None => {}
        }
    }

    /// The changed dot left of a setting at line `y`, or (row hovered) its reset arrow.
    fn draw_changed_mark(
        &mut self,
        painter: &mut Painter<'_>,
        area: &Area,
        y: f32,
        hover: bool,
        target: Target,
    ) {
        let s = area.s;
        let cy = y + OPT_H * s * 0.5;
        if hover && area.interactive {
            let r = Rect::new(area.x - 4.0 * s, cy - 8.0 * s, 16.0 * s, 16.0 * s);
            painter.round_rect(r, 5.0 * s, Fill::Solid(white(0.12)));
            draw_icon(painter, RowIcon::Refresh, r, s * 0.8);
            if let Some(hit) = area.hit(painter, r) {
                self.hits.push((hit, target));
                if hit.contains(self.cursor.0, self.cursor.1) {
                    self.set_tooltip("Back to its default", self.cursor);
                }
            }
        } else {
            painter.circle(area.x + 4.0 * s, cy, 6.0 * s, white(0.08));
            painter.circle(area.x + 4.0 * s, cy, 2.8 * s, white(0.92));
        }
    }

    /// One setting: its label (with the changed dot in the gutter) and its widget at the right,
    /// or a slider's full-width track under it; the colour editor opens below a colour.
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    fn draw_option_row(
        &mut self,
        painter: &mut Painter<'_>,
        env: &mut MenuEnv<'_>,
        m: usize,
        o: usize,
        area: &Area,
        y: f32,
        why: Why,
        reveal: &mut Option<(f32, f32)>,
    ) -> f32 {
        let s = area.s;
        let settings = env.settings;
        let descriptor = settings.catalog()[m];
        let option = &descriptor.options[o];
        let x0 = area.x + GUTTER * s;
        let inner_w = area.right() - x0;
        let h = option_h(option) * s;
        let editor_open = self.color_editor == Some((m, o));
        let full_h = h + if editor_open { COLOR_EDITOR_H * s } else { 0.0 };
        let full = Rect::new(area.x - 4.0 * s, y, area.w + 4.0 * s, full_h);
        let line = Rect::new(x0, y, inner_w, h);
        let value = settings
            .value_at(m, o)
            .cloned()
            .unwrap_or(OptionValue::Action);
        let changed = option.kind.is_persisted() && value != option.default;
        let hover = self.drag.is_none()
            && area
                .hit(painter, full)
                .is_some_and(|r| r.contains(self.cursor.0, self.cursor.1));
        let details = Pane::Details.index();
        let index = self.nav[details].len();
        let focused =
            self.keyboard && self.pane == Pane::Details && self.cursors[details] == Some(index);
        self.draw_flash(painter, env, o, full);
        if focused {
            draw_focus_ring(painter, full, 8.0 * s, s);
        }
        let saved_alpha = painter.alpha;
        if why.is_some() {
            painter.alpha *= 0.45;
        }
        if !matches!(option.kind, OptionKind::Button { .. }) {
            let capturing = self.capture == Some((m, o));
            let widget_w = option_widget_width(env, option, &value, capturing, inner_w, s);
            let room = (inner_w - widget_w - 10.0 * s).max(1.0);
            let mut style = TextStyle::new(12.5 * s);
            let natural = env.text.measure(option.label, style, env.mc_font);
            if natural > room {
                style.size = (style.size * room / natural).max(10.5 * s);
            }
            let (_, line_h) = env.text.metrics(style);
            shadowed(
                painter,
                env,
                option.label,
                x0,
                y + (OPT_H * s - line_h) * 0.5,
                style,
                white(0.86),
                s,
            );
            let label = Rect::new(x0 - 2.0 * s, y, natural.min(room) + 4.0 * s, OPT_H * s);
            if let Some(hit) = area.hit(painter, label) {
                self.hits.push((hit, Target::Label(m, o)));
                if hit.contains(self.cursor.0, self.cursor.1) {
                    self.option_tooltip(descriptor, option, why);
                }
            }
        }
        let primary = match option.kind {
            OptionKind::Toggle => Target::Toggle(m, o),
            OptionKind::Choice { .. } => Target::Choice(m, o),
            OptionKind::Keybind => Target::Key(m, o),
            OptionKind::Text { .. } => Target::Text(m, o),
            OptionKind::Button { .. } => Target::Button(m, o),
            OptionKind::Color { .. } => Target::Swatch(m, o),
            _ => Target::Label(m, o),
        };
        match option.kind {
            OptionKind::Toggle => {
                let on = matches!(value, OptionValue::Bool(true));
                let sw = Rect::new(
                    line.right() - 30.0 * s,
                    y + (OPT_H * s - 16.0 * s) * 0.5,
                    30.0 * s,
                    16.0 * s,
                );
                draw_switch(painter, sw, if on { 1.0 } else { 0.0 }, hover);
                if let Some(hit) = area.hit(painter, line) {
                    self.hits.push((hit, Target::Toggle(m, o)));
                }
            }
            OptionKind::Choice { .. } => {
                let text = option.kind.format_value(&value);
                let badge = draw_badge(
                    painter,
                    env,
                    &text,
                    line.right(),
                    y,
                    OPT_H * s,
                    s,
                    hover,
                    true,
                );
                if let Some(hit) = area.hit(painter, badge.inset(-3.0 * s)) {
                    self.hits.push((hit, Target::Choice(m, o)));
                }
            }
            OptionKind::Keybind => {
                let capturing = self.capture == Some((m, o));
                let text = if capturing {
                    "Press a key\u{2026}".to_owned()
                } else {
                    match value {
                        OptionValue::Key(code) => key_name(code),
                        _ => String::new(),
                    }
                };
                let badge = draw_badge(
                    painter,
                    env,
                    &text,
                    line.right(),
                    y,
                    OPT_H * s,
                    s,
                    hover || capturing,
                    false,
                );
                if capturing {
                    painter.round_rect_stroke(badge, 8.0 * s, 1.0, 0.5, white(0.85));
                }
                if let Some(hit) = area.hit(painter, badge.inset(-3.0 * s)) {
                    self.hits.push((hit, Target::Key(m, o)));
                }
            }
            OptionKind::Text { .. } => {
                let field_w = (inner_w * 0.55).max(80.0 * s);
                let field = Rect::new(
                    line.right() - field_w,
                    y + 5.0 * s,
                    field_w,
                    OPT_H * s - 10.0 * s,
                );
                self.draw_text_field(painter, env, m, o, field, "", area);
            }
            OptionKind::Button { label } => {
                let style = TextStyle::new(12.0 * s).weight(Weight::Semibold);
                let tw = env.text.measure(label, style, env.mc_font);
                let button = Rect::new(x0, y + 3.0 * s, tw + 28.0 * s, OPT_H * s - 6.0 * s);
                let hover = area
                    .hit(painter, button)
                    .is_some_and(|r| r.contains(self.cursor.0, self.cursor.1));
                painter.round_rect(
                    button,
                    10.0 * s,
                    Fill::Solid(white(if hover { 0.20 } else { 0.12 })),
                );
                painter.round_rect_stroke(button, 10.0 * s, 1.0, 0.5, white(0.32));
                let (_, lh) = env.text.metrics(style);
                shadowed(
                    painter,
                    env,
                    label,
                    button.x + 14.0 * s,
                    button.y + (button.h - lh) * 0.5,
                    style,
                    white(0.95),
                    s,
                );
                if let Some(hit) = area.hit(painter, button) {
                    self.hits.push((hit, Target::Button(m, o)));
                    if hit.contains(self.cursor.0, self.cursor.1) {
                        self.option_tooltip(descriptor, option, why);
                    }
                }
            }
            OptionKind::Color { alpha } => {
                let argb = match value {
                    OptionValue::Color(c) => c,
                    _ => 0xFFFF_FFFF,
                };
                let swatch = Rect::new(line.right() - 30.0 * s, y + 8.0 * s, 30.0 * s, 16.0 * s);
                draw_checker(painter, swatch, s);
                painter.round_rect(swatch, 5.0 * s, Fill::Solid(paint::argb(argb)));
                painter.round_rect_stroke(swatch, 5.0 * s, 1.0, 0.5, white(0.45));
                let hex = if alpha {
                    format!("#{argb:08X}")
                } else {
                    format!("#{:06X}", argb & 0x00FF_FFFF)
                };
                let style = TextStyle::new(10.5 * s);
                let tw = env.text.measure(&hex, style, env.mc_font);
                let (_, lh) = env.text.metrics(style);
                shadowed(
                    painter,
                    env,
                    &hex,
                    swatch.x - tw - 8.0 * s,
                    y + (OPT_H * s - lh) * 0.5,
                    style,
                    white(0.55),
                    s,
                );
                let pick = Rect::new(
                    swatch.x - tw - 10.0 * s,
                    y,
                    tw + swatch.w + 10.0 * s,
                    OPT_H * s,
                );
                if let Some(hit) = area.hit(painter, pick) {
                    self.hits.push((hit, Target::Swatch(m, o)));
                }
                if editor_open {
                    self.draw_color_editor(
                        painter,
                        env,
                        m,
                        o,
                        x0,
                        y + OPT_H * s,
                        inner_w,
                        s,
                        alpha,
                        argb,
                        area.clip,
                        area.interactive,
                    );
                }
            }
            OptionKind::Slider { min, max, .. } | OptionKind::SliderLabeled { min, max, .. } => {
                let v = match value {
                    OptionValue::Float(v) => v,
                    _ => min,
                };
                let text = option.kind.format_value(&value);
                draw_badge(
                    painter,
                    env,
                    &text,
                    line.right(),
                    y,
                    OPT_H * s,
                    s,
                    hover,
                    false,
                );
                let track = Rect::new(x0 + 2.0 * s, y + 34.0 * s, inner_w - 4.0 * s, 4.0 * s);
                let t = if max > min {
                    ((v - min) / (max - min)).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                draw_slider_track(painter, track, t, Fill::Solid(white(0.78)), s);
                let grab = Rect::new(
                    track.x - 6.0 * s,
                    track.y - 10.0 * s,
                    track.w + 12.0 * s,
                    track.h + 20.0 * s,
                );
                if let Some(hit) = area.hit(painter, grab) {
                    let (ox, oy) = painter.offset;
                    self.hits
                        .push((hit, Target::Slider(m, o, track.offset(ox, oy))));
                }
            }
            OptionKind::HiddenText { .. } => {}
        }
        if changed {
            self.draw_changed_mark(painter, area, y, hover, Target::ResetOption(m, o));
        }
        painter.alpha = saved_alpha;
        let span = area.span(y, y + full_h);
        self.nav[details].push(NavRow {
            enter: primary,
            space: matches!(option.kind, OptionKind::Toggle).then_some(primary),
            span,
        });
        if self.reveal_option == Some(o) || (focused && self.reveal_nav == Some(Pane::Details)) {
            *reveal = Some(span);
        }
        y + full_h
    }

    /// A text field for text option `o` of mod `m` (a long text shows its end, where the caret
    /// is); `placeholder` shows while it is empty and unfocused.
    #[allow(clippy::too_many_arguments)]
    fn draw_text_field(
        &mut self,
        painter: &mut Painter<'_>,
        env: &mut MenuEnv<'_>,
        m: usize,
        o: usize,
        field: Rect,
        placeholder: &str,
        area: &Area,
    ) {
        let s = area.s;
        let text = match env.settings.value_at(m, o) {
            Some(OptionValue::Text(text)) => text.as_ref(),
            _ => "",
        };
        let focused = self.text_focus == Some((m, o));
        painter.round_rect(field, 7.0 * s, Fill::Solid(black(0.25)));
        painter.round_rect_stroke(
            field,
            7.0 * s,
            1.0,
            0.5,
            white(if focused { 0.7 } else { 0.22 }),
        );
        let style = TextStyle::new(11.5 * s);
        let (_, lh) = env.text.metrics(style);
        // A text longer than the field (a chat macro) scrolls so its end, where the caret is,
        // stays in view.
        let room = field.w - 14.0 * s;
        let tw = env.text.measure(text, style, env.mc_font);
        let tx = field.x + 7.0 * s - (tw - room).max(0.0);
        let ty = field.y + (field.h - lh) * 0.5;
        let saved = painter.clip();
        painter.set_clip(Some(
            field.inset(2.0 * s).intersect(&saved.unwrap_or(field)),
        ));
        if text.is_empty() && !focused {
            if !placeholder.is_empty() {
                env.text.draw(
                    painter,
                    env.mc_font,
                    placeholder,
                    field.x + 7.0 * s,
                    ty,
                    style,
                    white(0.34),
                );
            }
        } else {
            let advance = env
                .text
                .draw(painter, env.mc_font, text, tx, ty, style, white(0.92));
            let blink = (ms_since(self.opened_at, env.now) as u64 / 500).is_multiple_of(2);
            if focused && blink {
                painter.rect(
                    Rect::new(
                        tx + advance + 1.0,
                        ty + 1.0 * s,
                        1.0 * s.max(1.0),
                        lh - 2.0 * s,
                    ),
                    white(0.9),
                );
            }
        }
        painter.set_clip(saved);
        if let Some(hit) = area.hit(painter, field) {
            self.hits.push((hit, Target::Text(m, o)));
        }
    }

    /// Two settings on one row (Chat Macros): the name ("Macro 1"), the key, then the text.
    #[allow(clippy::too_many_arguments)]
    fn draw_pair_row(
        &mut self,
        painter: &mut Painter<'_>,
        env: &mut MenuEnv<'_>,
        m: usize,
        (left, right): (usize, usize),
        area: &Area,
        y: f32,
        why: Why,
        reveal: &mut Option<(f32, f32)>,
    ) -> f32 {
        let s = area.s;
        let settings = env.settings;
        let descriptor = settings.catalog()[m];
        let (key_option, text_option) = (&descriptor.options[left], &descriptor.options[right]);
        let x0 = area.x + GUTTER * s;
        let inner_w = area.right() - x0;
        let h = OPT_H * s;
        let full = Rect::new(area.x - 4.0 * s, y, area.w + 4.0 * s, h);
        let hover = self.drag.is_none()
            && area
                .hit(painter, full)
                .is_some_and(|r| r.contains(self.cursor.0, self.cursor.1));
        let details = Pane::Details.index();
        let first = self.nav[details].len();
        let focused = self.keyboard
            && self.pane == Pane::Details
            && self.cursors[details].is_some_and(|c| c == first || c == first + 1);
        self.draw_flash(painter, env, left, full);
        self.draw_flash(painter, env, right, full);
        if focused {
            draw_focus_ring(painter, full, 8.0 * s, s);
        }
        let saved_alpha = painter.alpha;
        if why.is_some() {
            painter.alpha *= 0.45;
        }
        let name = key_option
            .label
            .strip_suffix(" key")
            .unwrap_or(key_option.label);
        let style = TextStyle::new(12.5 * s);
        let (_, lh) = env.text.metrics(style);
        let advance = shadowed(
            painter,
            env,
            name,
            x0,
            y + (h - lh) * 0.5,
            style,
            white(0.86),
            s,
        );
        if let Some(hit) = area.hit(painter, Rect::new(x0 - 2.0 * s, y, advance + 4.0 * s, h)) {
            self.hits.push((hit, Target::Label(m, left)));
            if hit.contains(self.cursor.0, self.cursor.1) {
                self.option_tooltip(descriptor, text_option, why);
            }
        }
        let field_w = (inner_w * 0.58).max(120.0 * s);
        let field = Rect::new(area.right() - field_w, y + 5.0 * s, field_w, h - 10.0 * s);
        self.draw_text_field(
            painter,
            env,
            m,
            right,
            field,
            "Chat message or /command",
            area,
        );
        let capturing = self.capture == Some((m, left));
        let key_text = if capturing {
            "Press a key\u{2026}".to_owned()
        } else {
            match settings.value_at(m, left) {
                Some(OptionValue::Key(code)) => key_name(*code),
                _ => String::new(),
            }
        };
        let badge = draw_badge(
            painter,
            env,
            &key_text,
            field.x - 8.0 * s,
            y,
            h,
            s,
            hover || capturing,
            false,
        );
        if capturing {
            painter.round_rect_stroke(badge, 8.0 * s, 1.0, 0.5, white(0.85));
        }
        if let Some(hit) = area.hit(painter, badge.inset(-3.0 * s)) {
            self.hits.push((hit, Target::Key(m, left)));
        }
        let changed = [left, right].iter().any(|&o| {
            settings
                .value_at(m, o)
                .is_some_and(|value| *value != descriptor.options[o].default)
        });
        if changed {
            self.draw_changed_mark(painter, area, y, hover, Target::ResetOption(m, left));
        }
        painter.alpha = saved_alpha;
        let span = area.span(y, y + h);
        for enter in [Target::Key(m, left), Target::Text(m, right)] {
            self.nav[details].push(NavRow {
                enter,
                space: None,
                span,
            });
        }
        if [Some(left), Some(right)].contains(&self.reveal_option)
            || (focused && self.reveal_nav == Some(Pane::Details))
        {
            *reveal = Some(span);
        }
        y + h
    }

    /// A choice drawn as tabs (Particles' "Particle"), then the settings that follow the picked
    /// tab, in a panel under it.
    #[allow(clippy::too_many_arguments)]
    fn draw_tabs(
        &mut self,
        painter: &mut Painter<'_>,
        env: &mut MenuEnv<'_>,
        m: usize,
        picker: usize,
        edits: &[usize],
        area: &Area,
        y: f32,
        why: Why,
        reveal: &mut Option<(f32, f32)>,
    ) -> f32 {
        let s = area.s;
        let settings = env.settings;
        let descriptor = settings.catalog()[m];
        let option = &descriptor.options[picker];
        let OptionKind::Choice { choices } = option.kind else {
            return self.draw_option_row(painter, env, m, picker, area, y, why, reveal);
        };
        let current = match settings.value_at(m, picker) {
            Some(OptionValue::Choice(index)) => *index,
            _ => 0,
        };
        let x0 = area.x + GUTTER * s;
        let details = Pane::Details.index();
        let index = self.nav[details].len();
        let focused =
            self.keyboard && self.pane == Pane::Details && self.cursors[details] == Some(index);
        let label_style = TextStyle::new(12.5 * s);
        let (_, lh) = env.text.metrics(label_style);
        shadowed(
            painter,
            env,
            option.label,
            x0,
            y + (26.0 * s - lh) * 0.5,
            label_style,
            white(0.86),
            s,
        );
        if let Some(hit) = area.hit(painter, Rect::new(x0, y, area.right() - x0, 26.0 * s)) {
            self.hits.push((hit, Target::Label(m, picker)));
            if hit.contains(self.cursor.0, self.cursor.1) {
                self.option_tooltip(descriptor, option, why);
            }
        }
        let tab_style = TextStyle::new(11.5 * s).weight(Weight::Semibold);
        let (_, th) = env.text.metrics(tab_style);
        let tab_h = th + 10.0 * s;
        let mut tab_y = y + 28.0 * s;
        let mut x = x0;
        let tabs_top = tab_y;
        for (i, label) in choices.iter().enumerate() {
            let tw = env.text.measure(label, tab_style, env.mc_font);
            let w = tw + 20.0 * s;
            if x + w > area.right() && x > x0 {
                x = x0;
                tab_y += tab_h + 6.0 * s;
            }
            let r = Rect::new(x, tab_y, w, tab_h);
            let hit = area.hit(painter, r);
            let hover = hit.is_some_and(|hit| hit.contains(self.cursor.0, self.cursor.1));
            let on = i == current;
            let (fill, stroke) = if on {
                (white(0.26), white(0.62))
            } else if hover {
                (white(0.14), white(0.30))
            } else {
                (black(0.18), white(0.16))
            };
            painter.round_rect(r, tab_h * 0.5, Fill::Solid(fill));
            painter.round_rect_stroke(r, tab_h * 0.5, 1.0, 0.5, stroke);
            shadowed(
                painter,
                env,
                label,
                r.x + 10.0 * s,
                r.y + 5.0 * s,
                tab_style,
                white(if on { 1.0 } else { 0.78 }),
                s,
            );
            if let Some(hit) = hit {
                self.hits.push((hit, Target::Tab(m, picker, i)));
            }
            x += w + 6.0 * s;
        }
        let tabs_bottom = tab_y + tab_h;
        if focused {
            draw_focus_ring(
                painter,
                Rect::new(
                    x0 - 4.0 * s,
                    tabs_top - 2.0 * s,
                    area.right() - x0 + 4.0 * s,
                    tabs_bottom - tabs_top + 4.0 * s,
                ),
                10.0 * s,
                s,
            );
        }
        let span = area.span(y, tabs_bottom);
        self.nav[details].push(NavRow {
            enter: Target::Choice(m, picker),
            space: None,
            span,
        });
        if self.reveal_option == Some(picker) || (focused && self.reveal_nav == Some(Pane::Details))
        {
            *reveal = Some(span);
        }
        // The settings the picked tab edits, in a panel under the tabs.
        let rows: Vec<(usize, Why)> = edits
            .iter()
            .copied()
            .filter(|&o| descriptor.options[o].kind.is_visible())
            .filter_map(|o| self.option_shown(settings, m, o).map(|why| (o, why)))
            .collect();
        let mut cy = tabs_bottom + 8.0 * s;
        if !rows.is_empty() {
            let panel_h: f32 = rows
                .iter()
                .map(|&(o, _)| option_h(&descriptor.options[o]) * s)
                .sum::<f32>()
                + 10.0 * s;
            let panel = Rect::new(area.x + 6.0 * s, cy, area.w - 6.0 * s, panel_h);
            painter.round_rect(panel, 10.0 * s, Fill::Solid(black(0.14)));
            painter.round_rect_stroke(panel, 10.0 * s, 1.0, 0.5, white(0.08));
            let inner = Area {
                x: area.x + 10.0 * s,
                w: area.w - 16.0 * s,
                ..*area
            };
            cy += 5.0 * s;
            for (o, why) in rows {
                cy = self.draw_option_row(painter, env, m, o, &inner, cy, why, reveal);
            }
            cy += 5.0 * s;
        }
        cy + 4.0 * s
    }

    /// A small text button whose right edge is at `right`, vertically centred in an
    /// [`OPT_H`] line at `y`. `enabled == false` draws it greyed out. Returns its rectangle.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn draw_small_button(
        &self,
        painter: &mut Painter<'_>,
        env: &mut MenuEnv<'_>,
        label: &str,
        (right, y): (f32, f32),
        s: f32,
        enabled: bool,
        armed: bool,
    ) -> Rect {
        let style = TextStyle::new(11.0 * s).weight(Weight::Semibold);
        let tw = env.text.measure(label, style, env.mc_font);
        let (_, lh) = env.text.metrics(style);
        let h = 20.0 * s;
        let r = Rect::new(
            right - tw - 18.0 * s,
            y + (OPT_H * s - h) * 0.5,
            tw + 18.0 * s,
            h,
        );
        let hover = enabled
            && r.offset(painter.offset.0, painter.offset.1)
                .contains(self.cursor.0, self.cursor.1);
        let fill = if armed {
            css(255, 110, 90, if hover { 0.42 } else { 0.30 })
        } else if !enabled {
            white(0.04)
        } else {
            white(if hover { 0.20 } else { 0.10 })
        };
        painter.round_rect(r, 7.0 * s, Fill::Solid(fill));
        let border = if armed {
            css(255, 150, 130, 0.8)
        } else {
            white(if enabled { 0.30 } else { 0.12 })
        };
        painter.round_rect_stroke(r, 7.0 * s, 1.0, 0.5, border);
        shadowed(
            painter,
            env,
            label,
            r.x + 9.0 * s,
            r.y + (r.h - lh) * 0.5,
            style,
            white(if enabled { 0.95 } else { 0.40 }),
            s,
        );
        r
    }

    /// "Reset to defaults" for mod `m`, right-aligned at `right` in the line at `y`. Greyed
    /// out while every option is at its default; the first click arms it ("Click to confirm",
    /// red) for `ARM_MS`, the second resets.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn draw_reset_button(
        &mut self,
        painter: &mut Painter<'_>,
        env: &mut MenuEnv<'_>,
        m: usize,
        (right, y): (f32, f32),
        s: f32,
        clip: Rect,
        interactive: bool,
    ) {
        let enabled = !env.settings.options_are_default_at(m);
        let armed = enabled
            && self
                .reset_armed
                .is_some_and(|(armed_m, at)| armed_m == m && still_armed(at, env.now));
        let label = if armed {
            "Click to confirm"
        } else {
            "Reset to defaults"
        };
        let r = self.draw_small_button(painter, env, label, (right, y), s, enabled, armed);
        if interactive && enabled {
            let off = painter.offset;
            let hr = r.offset(off.0, off.1).intersect(&clip.offset(off.0, off.1));
            if hr.w > 0.0 && hr.h > 0.0 {
                self.hits.push((hr, Target::ResetMod(m)));
            }
        }
    }

    /// The profiles block of mod `m` (colour grading) at `(x, y)`, `w` wide: a "Profiles"
    /// header (with "Reset to defaults" on its right when `resettable`), the name field with
    /// Save, then one row per profile (Load, Delete). Returns the y below it. The profile
    /// matching the current values is marked with a dot.
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    pub(super) fn draw_profiles(
        &mut self,
        painter: &mut Painter<'_>,
        env: &mut MenuEnv<'_>,
        m: usize,
        (x, y, w): (f32, f32, f32),
        s: f32,
        clip: Rect,
        interactive: bool,
        resettable: bool,
    ) -> f32 {
        let off = painter.offset;
        let clip_screen = clip.offset(off.0, off.1);
        let push_hit = |hits: &mut Vec<(Rect, Target)>, rect: Rect, target: Target| {
            if interactive {
                let r = rect.offset(off.0, off.1).intersect(&clip_screen);
                if r.w > 0.0 && r.h > 0.0 {
                    hits.push((r, target));
                }
            }
        };
        let mut cy = y;
        // The label and how many slots are used, like a small section title; the reset on the
        // right.
        let caps = TextStyle::new(9.5 * s).weight(Weight::Bold).caps(0.14);
        let (_, caps_h) = env.text.metrics(caps);
        let label_y = cy + (PROFILE_HEAD_H * s - caps_h) * 0.5;
        shadowed(painter, env, "Profiles", x, label_y, caps, white(0.62), s);
        let count = env.settings.profiles_at(m).len();
        let used = format!("{count}/{}", mc_mods::MAX_PROFILES);
        let label_w = env.text.measure("Profiles", caps, env.mc_font);
        shadowed(
            painter,
            env,
            &used,
            x + label_w + 8.0 * s,
            label_y,
            caps,
            white(0.40),
            s,
        );
        if resettable {
            let line_y = cy + (PROFILE_HEAD_H - OPT_H) * 0.5 * s;
            self.draw_reset_button(painter, env, m, (x + w, line_y), s, clip, interactive);
        }
        cy += PROFILE_HEAD_H * s;

        // The name field and Save (Update when the name is taken; greyed when full).
        let existing = env.settings.profile_named_at(m, &self.profile_name);
        let full = count >= mc_mods::MAX_PROFILES && existing.is_none();
        let save_label = if existing.is_some() { "Update" } else { "Save" };
        let save = self.draw_small_button(painter, env, save_label, (x + w, cy), s, !full, false);
        if !full {
            push_hit(&mut self.hits, save, Target::ProfileSave(m));
        }
        let field = Rect::new(x, cy + 5.0 * s, save.x - x - 8.0 * s, OPT_H * s - 10.0 * s);
        let focused = self.name_focus == Some(m);
        painter.round_rect(field, 7.0 * s, Fill::Solid(black(0.25)));
        painter.round_rect_stroke(
            field,
            7.0 * s,
            1.0,
            0.5,
            white(if focused { 0.7 } else { 0.22 }),
        );
        let style = TextStyle::new(11.5 * s);
        let (_, lh) = env.text.metrics(style);
        // Long names scroll so the caret stays in view, like a text option.
        let name_w = env.text.measure(&self.profile_name, style, env.mc_font);
        let tx = field.x + 7.0 * s - (name_w - (field.w - 14.0 * s)).max(0.0);
        let ty = field.y + (field.h - lh) * 0.5;
        let saved_clip = painter.clip();
        painter.set_clip(Some(
            field.inset(2.0 * s).intersect(&saved_clip.unwrap_or(field)),
        ));
        if self.profile_name.is_empty() && !focused {
            env.text.draw(
                painter,
                env.mc_font,
                "New profile name",
                tx,
                ty,
                style,
                white(0.38),
            );
        } else {
            let advance = env.text.draw(
                painter,
                env.mc_font,
                &self.profile_name,
                tx,
                ty,
                style,
                white(0.92),
            );
            let blink = (ms_since(self.opened_at, env.now) as u64 / 500).is_multiple_of(2);
            if focused && blink {
                painter.rect(
                    Rect::new(
                        tx + advance + 1.0,
                        ty + 1.0 * s,
                        1.0 * s.max(1.0),
                        lh - 2.0 * s,
                    ),
                    white(0.9),
                );
            }
        }
        painter.set_clip(saved_clip);
        push_hit(&mut self.hits, field, Target::ProfileName(m));
        cy += OPT_H * s;

        // One row per profile.
        if count == 0 {
            let note = TextStyle::new(11.0 * s);
            let (_, nh) = env.text.metrics(note);
            shadowed(
                painter,
                env,
                "No profiles yet: set the sliders, then Save.",
                x,
                cy + (DYN_H * s - nh) * 0.5,
                note,
                white(0.45),
                s,
            );
            return cy + DYN_H * s;
        }
        let row_style = TextStyle::new(12.0 * s);
        let (_, rh) = env.text.metrics(row_style);
        for p in 0..count {
            let current = env.settings.profile_is_current_at(m, p);
            let armed = self
                .delete_armed
                .is_some_and(|(am, ap, at)| (am, ap) == (m, p) && still_armed(at, env.now));
            // Delete (×; "Delete?" while armed), then Load, from the right.
            let delete = if armed {
                self.draw_small_button(painter, env, "Delete?", (x + w, cy), s, true, true)
            } else {
                let r = Rect::new(
                    x + w - 22.0 * s,
                    cy + (DYN_H * s - 20.0 * s) * 0.5,
                    22.0 * s,
                    20.0 * s,
                );
                let hover = interactive
                    && r.offset(off.0, off.1)
                        .contains(self.cursor.0, self.cursor.1);
                painter.round_rect(
                    r,
                    7.0 * s,
                    Fill::Solid(white(if hover { 0.18 } else { 0.08 })),
                );
                painter.round_rect_stroke(r, 7.0 * s, 1.0, 0.5, white(0.22));
                draw_icon(painter, RowIcon::Delete, r, s);
                r
            };
            push_hit(&mut self.hits, delete, Target::ProfileDelete(m, p));
            let at = (delete.x - 6.0 * s, cy);
            let load = self.draw_small_button(painter, env, "Load", at, s, true, false);
            push_hit(&mut self.hits, load, Target::ProfileLoad(m, p));
            // The name, with a dot when it is what the sliders show now.
            let dot_x = x + 4.0 * s;
            let dot_y = cy + DYN_H * s * 0.5;
            if current {
                painter.circle(dot_x, dot_y, 3.0 * s, white(0.95));
            } else {
                painter.round_rect_stroke(
                    Rect::new(dot_x - 3.0 * s, dot_y - 3.0 * s, 6.0 * s, 6.0 * s),
                    3.0 * s,
                    1.0 * s,
                    0.5 * s,
                    white(0.35),
                );
            }
            let name_x = x + 14.0 * s;
            let saved_clip = painter.clip();
            let name_clip = Rect::new(name_x, cy, (load.x - 6.0 * s - name_x).max(0.0), DYN_H * s);
            painter.set_clip(Some(
                saved_clip.map_or(name_clip, |c| c.intersect(&name_clip)),
            ));
            let name = env.settings.profiles_at(m)[p].name().to_owned();
            shadowed(
                painter,
                env,
                &name,
                name_x,
                cy + (DYN_H * s - rh) * 0.5,
                row_style,
                white(if current { 0.97 } else { 0.80 }),
                s,
            );
            painter.set_clip(saved_clip);
            cy += DYN_H * s;
        }
        cy
    }

    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    pub(super) fn draw_color_editor(
        &mut self,
        painter: &mut Painter<'_>,
        env: &mut MenuEnv<'_>,
        m: usize,
        o: usize,
        x: f32,
        y: f32,
        w: f32,
        s: f32,
        alpha: bool,
        argb: u32,
        clip: Rect,
        interactive: bool,
    ) -> f32 {
        let off = painter.offset;
        let clip_screen = clip.offset(off.0, off.1);
        // Keep the editor's HSV in step with outside changes (another widget, a reset).
        if hsva_to_argb(self.color_hsv) != argb {
            self.color_hsv = argb_to_hsva(argb);
        }
        let mut cy = y + 2.0 * s;
        // Presets.
        let size = 16.0 * s;
        let gap = ((w - size * PRESETS.len() as f32) / (PRESETS.len() as f32 - 1.0)).max(2.0 * s);
        for (i, preset) in PRESETS.iter().enumerate() {
            let r = Rect::new(x + i as f32 * (size + gap), cy, size, size);
            painter.round_rect(r, size * 0.5, Fill::Solid(paint::argb(*preset)));
            let selected = (argb & 0x00FF_FFFF) == (preset & 0x00FF_FFFF);
            painter.round_rect_stroke(
                r,
                size * 0.5,
                1.0,
                0.5,
                white(if selected { 0.95 } else { 0.35 }),
            );
            if interactive {
                let hr = r.offset(off.0, off.1).intersect(&clip_screen);
                if hr.w > 0.0 {
                    self.hits.push((hr, Target::Preset(m, o, *preset)));
                }
            }
        }
        cy += size + 10.0 * s;
        let [h, sat, v, a] = self.color_hsv;
        let channels: &[(u8, &str)] = if alpha {
            &[(0, "H"), (1, "S"), (2, "V"), (3, "A")]
        } else {
            &[(0, "H"), (1, "S"), (2, "V")]
        };
        let label_style = TextStyle::new(10.5 * s).weight(Weight::Semibold);
        for &(channel, name) in channels {
            let (_, lh) = env.text.metrics(label_style);
            shadowed(
                painter,
                env,
                name,
                x,
                cy + (12.0 * s - lh) * 0.5,
                label_style,
                white(0.6),
                s,
            );
            let track = Rect::new(x + 16.0 * s, cy + 3.0 * s, w - 18.0 * s, 6.0 * s);
            match channel {
                0 => {
                    // The hue rainbow in six segments.
                    for seg in 0..6 {
                        let t0 = seg as f32 / 6.0;
                        let t1 = (seg + 1) as f32 / 6.0;
                        let r = Rect::new(
                            track.x + track.w * t0,
                            track.y,
                            track.w / 6.0 + 0.5,
                            track.h,
                        );
                        let c0 = paint::argb(hsva_to_argb([t0, 1.0, 1.0, 1.0]));
                        let c1 = paint::argb(hsva_to_argb([t1, 1.0, 1.0, 1.0]));
                        painter.round_rect(
                            r,
                            0.0,
                            Fill::Linear {
                                from: (r.x, r.y),
                                to: (r.right(), r.y),
                                c0,
                                c1,
                            },
                        );
                    }
                }
                1 => {
                    let c0 = paint::argb(hsva_to_argb([h, 0.0, v, 1.0]));
                    let c1 = paint::argb(hsva_to_argb([h, 1.0, v, 1.0]));
                    painter.round_rect(
                        track,
                        track.h * 0.5,
                        Fill::Linear {
                            from: (track.x, track.y),
                            to: (track.right(), track.y),
                            c0,
                            c1,
                        },
                    );
                }
                2 => {
                    let c1 = paint::argb(hsva_to_argb([h, sat, 1.0, 1.0]));
                    painter.round_rect(
                        track,
                        track.h * 0.5,
                        Fill::Linear {
                            from: (track.x, track.y),
                            to: (track.right(), track.y),
                            c0: black(1.0),
                            c1,
                        },
                    );
                }
                _ => {
                    draw_checker(painter, track, s);
                    let mut c1 = paint::argb(hsva_to_argb([h, sat, v, 1.0]));
                    let mut c0 = c1;
                    c0[3] = 0.0;
                    c1[3] = 1.0;
                    painter.round_rect(
                        track,
                        track.h * 0.5,
                        Fill::Linear {
                            from: (track.x, track.y),
                            to: (track.right(), track.y),
                            c0,
                            c1,
                        },
                    );
                }
            }
            let t = [h, sat, v, a][usize::from(channel)];
            painter.circle(
                track.x + track.w * t,
                track.y + track.h * 0.5,
                6.0 * s,
                white(0.96),
            );
            painter.round_rect_stroke(
                Rect::new(
                    track.x + track.w * t - 6.0 * s,
                    track.y + track.h * 0.5 - 6.0 * s,
                    12.0 * s,
                    12.0 * s,
                ),
                6.0 * s,
                1.0,
                0.5,
                black(0.35),
            );
            if interactive {
                let grab = Rect::new(
                    track.x - 6.0 * s,
                    cy - 2.0 * s,
                    track.w + 12.0 * s,
                    16.0 * s,
                );
                let hr = grab.offset(off.0, off.1).intersect(&clip_screen);
                if hr.w > 0.0 {
                    self.hits.push((
                        hr,
                        Target::ColorSlider(m, o, channel, track.offset(off.0, off.1)),
                    ));
                }
            }
            cy += 20.0 * s;
        }
        y + COLOR_EDITOR_H * s
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn draw_dynamic_row(
        &mut self,
        painter: &mut Painter<'_>,
        env: &mut MenuEnv<'_>,
        m: usize,
        row: &mc_mods::DynamicRow,
        x: f32,
        y: f32,
        w: f32,
        s: f32,
        clip: Rect,
        interactive: bool,
    ) -> f32 {
        let off = painter.offset;
        let clip_screen = clip.offset(off.0, off.1);
        let h = (DYN_H + if row.level.is_some() { 8.0 } else { 0.0 }) * s;
        let style = TextStyle::new(12.5 * s);
        let (_, lh) = env.text.metrics(style);
        let alpha = if row.muted { 0.45 } else { 0.85 };
        let mut right = x + w;
        // Buttons from the right.
        for (index, button) in row.buttons.iter().enumerate().rev() {
            let Some(button) = button else { continue };
            let bw = match button {
                RowButton::Icon(_) => 22.0 * s,
                RowButton::Text(label) => {
                    env.text
                        .measure(label, TextStyle::new(11.0 * s), env.mc_font)
                        + 16.0 * s
                }
            };
            let r = Rect::new(right - bw, y + (DYN_H * s - 20.0 * s) * 0.5, bw, 20.0 * s);
            let hover = interactive
                && r.offset(off.0, off.1)
                    .contains(self.cursor.0, self.cursor.1);
            painter.round_rect(
                r,
                7.0 * s,
                Fill::Solid(white(if hover { 0.18 } else { 0.08 })),
            );
            painter.round_rect_stroke(r, 7.0 * s, 1.0, 0.5, white(0.22));
            match button {
                RowButton::Icon(icon) => draw_icon(painter, *icon, r, s),
                RowButton::Text(label) => {
                    let st = TextStyle::new(11.0 * s);
                    let (_, blh) = env.text.metrics(st);
                    shadowed(
                        painter,
                        env,
                        label,
                        r.x + 8.0 * s,
                        r.y + (r.h - blh) * 0.5,
                        st,
                        white(0.9),
                        s,
                    );
                }
            }
            if interactive {
                let hr = r.offset(off.0, off.1).intersect(&clip_screen);
                if hr.w > 0.0 {
                    self.hits
                        .push((hr, Target::DynAction(m, row.key, index as u8)));
                }
            }
            right = r.x - 6.0 * s;
        }
        // The badge (a clickable choice when `cycle`).
        if !row.badge.is_empty() {
            let hover = interactive && row.cycle;
            let badge = draw_badge(
                painter,
                env,
                &row.badge,
                right,
                y,
                DYN_H * s,
                s,
                false,
                row.cycle,
            );
            if hover {
                let hr = badge
                    .inset(-3.0 * s)
                    .offset(off.0, off.1)
                    .intersect(&clip_screen);
                if hr.w > 0.0 {
                    self.hits.push((hr, Target::DynCycle(m, row.key)));
                }
            }
            right = badge.x - 6.0 * s;
        }
        // The label, cut at the space left.
        let saved = painter.clip();
        let label_clip = Rect::new(x, y, (right - x).max(0.0), h);
        painter.set_clip(Some(saved.map_or(label_clip, |c| c.intersect(&label_clip))));
        shadowed(
            painter,
            env,
            &row.label,
            x,
            y + (DYN_H * s - lh) * 0.5,
            style,
            white(alpha),
            s,
        );
        painter.set_clip(saved);
        if let Some(level) = row.level {
            let track = Rect::new(x, y + DYN_H * s - 2.0 * s, w, 4.0 * s);
            draw_slider_track(
                painter,
                track,
                level.clamp(0.0, 1.0),
                Fill::Linear {
                    from: (track.x, track.y),
                    to: (track.right(), track.y),
                    c0: css(120, 255, 170, 0.9),
                    c1: css(255, 110, 90, 0.95),
                },
                s,
            );
        }
        y + h
    }
}

/// How wide an option's widget is at the right of its line (what `draw_option_row` draws for
/// it), so the label can stop short of it.
pub(super) fn option_widget_width(
    env: &mut MenuEnv<'_>,
    option: &mc_mods::OptionDescriptor,
    value: &OptionValue,
    capturing: bool,
    inner_w: f32,
    s: f32,
) -> f32 {
    let badge_style = TextStyle::new(11.0 * s);
    let badge = |text: &str, arrows: bool| {
        env.text.measure(text, badge_style, env.mc_font)
            + 16.0 * s
            + if arrows { 24.0 * s } else { 0.0 }
    };
    match option.kind {
        OptionKind::Toggle => 30.0 * s,
        OptionKind::Choice { .. } => badge(&option.kind.format_value(value), true),
        OptionKind::Slider { .. } | OptionKind::SliderLabeled { .. } => {
            badge(&option.kind.format_value(value), false)
        }
        OptionKind::Keybind => match (capturing, value) {
            (true, _) => badge("Press a key\u{2026}", false),
            (false, OptionValue::Key(code)) => badge(&key_name(*code), false),
            _ => 0.0,
        },
        OptionKind::Text { .. } => (inner_w * 0.55).max(80.0 * s),
        OptionKind::Color { alpha } => {
            let hex = if alpha { "#FFFFFFFF" } else { "#FFFFFF" };
            env.text.measure(hex, TextStyle::new(10.5 * s), env.mc_font) + 38.0 * s
        }
        OptionKind::Button { .. } | OptionKind::HiddenText { .. } => 0.0,
    }
}
