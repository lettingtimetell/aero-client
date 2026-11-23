//! Key names, and the Keybinds view.
//!
//! * **Window 1:** how many keys the mods use, with an amber "!" while two keys that work
//!   right now share a key ([`ModMenuScreen::keybinds_badge`]).
//! * **Window 2:** one list of every key: the menu key, then each mod's keys in menu order. A
//!   click rebinds through the shell's capture (Esc cancels, Backspace unbinds, a mouse button
//!   binds it; the menu key is keyboard-only and Backspace puts Right Shift back); a right click
//!   puts the default back. A key that shares its key with another mod key, the menu key or a
//!   Minecraft control from `options.txt` says so under it ("Also: Drop Item"), in amber when
//!   both work right now. Keys of mods that are off are dimmed: they do nothing until the mod
//!   is on.
//! * **Window 3:** Minecraft's own controls, read-only, with the ones a mod key also uses
//!   marked, so a free key is easy to find (they change in Options › Controls).
//!
//! The shell's hooks: [`ModMenuScreen::keybinds_badge`], [`ModMenuScreen::draw_keybinds_list`],
//! [`ModMenuScreen::draw_keybinds_details`], [`KeysTarget`] and
//! [`ModMenuScreen::activate_keys`]. The rows also push the shell's own `Target::Key` and
//! `Target::MenuKey` hits under theirs, so `menu_key_rect` and the capture keep working.

use mc_mods::menu_rules;
use mc_mods::{ModSettings, OptionKind, OptionValue};
use mc_render::input::KeyBindings;

use super::layout::{draw_title, Area};
use super::paint::{css, white, Fill, Painter, Rect, Rgba};
use super::sidebar::NavBadge;
use super::state::{NavRow, Pane, Target};
use super::text::{TextStyle, Weight};
use super::widgets::{draw_badge, draw_focus_ring, draw_wrapped, shadowed};
use super::{MenuEnv, MenuInput, ModMenuScreen};

/// A key code as the menu shows it: LWJGL's names (as vanilla's Controls screen), with the
/// modifier keys spelled out ("RMENU" is Right Alt).
#[must_use]
pub fn key_name(code: i32) -> String {
    let name = match code {
        0 => "None",
        29 => "Left Ctrl",
        42 => "Left Shift",
        54 => "Right Shift",
        56 => "Left Alt",
        157 => "Right Ctrl",
        184 => "Right Alt",
        219 => "Left Win",
        220 => "Right Win",
        _ => return mc_render::input::Input::from_lwjgl(code).display_name(),
    };
    name.to_owned()
}

/// Keys vanilla handles itself, outside the controls in `options.txt`: (LWJGL code, name).
/// F4 is `Minecraft.runTick`'s `switchUseShader` (`k == 62`, in game): a mod key
/// on F4 also turned a Super Secret Settings shader off and on, with no warning.
pub(super) const FIXED_KEYS: [(i32, &str); 3] = [
    (59, "Hide the HUD (F1)"),
    (61, "Debug screen (F3)"),
    (62, "Shader on/off (F4)"),
];

/// The amber of a clash.
const CLASH: Rgba = css(255, 196, 82, 0.95);
/// A key row's first line.
const ROW_H: f32 = 34.0;
/// A key row with its "Also: ..." line.
const ROW_NOTE_H: f32 = 50.0;

/// A vanilla control's English name (its `en_US.lang` line), from its translation key.
#[must_use]
pub(super) fn control_name(description: &str) -> String {
    let name = match description {
        "key.attack" => "Attack/Destroy",
        "key.use" => "Use Item/Place Block",
        "key.forward" => "Walk Forwards",
        "key.left" => "Strafe Left",
        "key.back" => "Walk Backwards",
        "key.right" => "Strafe Right",
        "key.jump" => "Jump",
        "key.sneak" => "Sneak",
        "key.sprint" => "Sprint",
        "key.drop" => "Drop Item",
        "key.inventory" => "Inventory",
        "key.chat" => "Open Chat",
        "key.playerlist" => "List Players",
        "key.pickItem" => "Pick Block",
        "key.command" => "Open Command",
        "key.screenshot" => "Take Screenshot",
        "key.togglePerspective" => "Toggle Perspective",
        "key.smoothCamera" => "Toggle Cinematic Camera",
        "key.fullscreen" => "Toggle Fullscreen",
        "key.spectatorOutlines" => "Highlight Players (Spectators)",
        "key.streamStartStop" => "Start/Stop Stream",
        "key.streamPauseUnpause" => "Pause/Unpause Stream",
        "key.streamCommercial" => "Show Stream Commercials",
        "key.streamToggleMic" => "Push To Talk/Mute",
        other => {
            return match other.strip_prefix("key.hotbar.") {
                Some(slot) => format!("Hotbar Slot {slot}"),
                None => other.to_owned(),
            };
        }
    };
    name.to_owned()
}

/// A vanilla control category's English name.
fn category_name(category: &str) -> &str {
    match category {
        "key.categories.movement" => "Movement",
        "key.categories.gameplay" => "Gameplay",
        "key.categories.inventory" => "Inventory",
        "key.categories.multiplayer" => "Multiplayer",
        "key.categories.misc" => "Miscellaneous",
        "key.categories.stream" => "Streaming",
        "key.categories.ui" => "Game Interface",
        other => other,
    }
}

/// Who a key belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Owner {
    /// The menu key.
    Menu,
    /// Key setting `.1` of mod `.0`.
    Mod(usize, usize),
}

/// One key a mod uses, or the menu key.
#[derive(Debug, Clone, Copy)]
struct ModKey {
    owner: Owner,
    /// LWJGL encoding: `> 0` a key, `< 0` a mouse button, `0` unbound.
    code: i32,
    /// It works right now: the menu key, or a key of a mod that is on (or that works while its
    /// mod is off, `menu_rules::applies_while_off`).
    active: bool,
}

/// Every key: the menu key first, then each key setting of the catalog in catalog order.
fn mod_keys(settings: &ModSettings) -> impl Iterator<Item = ModKey> + '_ {
    let menu = std::iter::once(ModKey {
        owner: Owner::Menu,
        code: settings.menu_key(),
        active: true,
    });
    let mods = settings
        .catalog()
        .iter()
        .enumerate()
        .flat_map(move |(m, descriptor)| {
            descriptor
                .options
                .iter()
                .enumerate()
                .filter(|(_, option)| matches!(option.kind, OptionKind::Keybind))
                .map(move |(o, option)| ModKey {
                    owner: Owner::Mod(m, o),
                    code: match settings.value_at(m, o) {
                        Some(OptionValue::Key(code)) => *code,
                        _ => 0,
                    },
                    active: settings.enabled_at(m)
                        || menu_rules::applies_while_off(descriptor.id, option.id),
                })
        });
    menu.chain(mods)
}

/// Whether bound key `k` shares its key with something that works right now (another active
/// key, a Minecraft control, a fixed vanilla key).
fn active_clash(settings: &ModSettings, vanilla: Option<&KeyBindings>, k: ModKey) -> bool {
    if !k.active || k.code == 0 {
        return false;
    }
    mod_keys(settings).any(|other| other.owner != k.owner && other.active && other.code == k.code)
        || vanilla.is_some_and(|keys| keys.iter().any(|b| b.binding().to_lwjgl() == k.code))
        || FIXED_KEYS.iter().any(|(code, _)| *code == k.code)
}

/// What else uses `k`'s key, as names ("Drop Item", "Zoom key (Zoom)"), and whether any of
/// them clashes right now.
fn clash_names(
    settings: &ModSettings,
    vanilla: Option<&KeyBindings>,
    all: &[ModKey],
    k: ModKey,
) -> (Vec<String>, bool) {
    let mut names = Vec::new();
    let mut hot = false;
    if k.code == 0 {
        return (names, hot);
    }
    for other in all
        .iter()
        .filter(|other| other.owner != k.owner && other.code == k.code)
    {
        names.push(owner_name(settings, other.owner));
        hot |= k.active && other.active;
    }
    for binding in vanilla
        .into_iter()
        .flat_map(KeyBindings::iter)
        .filter(|b| b.binding().to_lwjgl() == k.code)
    {
        names.push(control_name(binding.description()));
        hot |= k.active;
    }
    for (_, name) in FIXED_KEYS.iter().filter(|(code, _)| *code == k.code) {
        names.push((*name).to_owned());
        hot |= k.active;
    }
    (names, hot)
}

/// A key's owner in words: "Open the menu", "Zoom key (Zoom)".
fn owner_name(settings: &ModSettings, owner: Owner) -> String {
    match owner {
        Owner::Menu => "Open the menu".to_owned(),
        Owner::Mod(m, o) => {
            let descriptor = settings.catalog()[m];
            format!("{} ({})", descriptor.options[o].label, descriptor.name)
        }
    }
}

/// A section title like window 2's: small bold spaced capitals.
fn draw_section(
    painter: &mut Painter<'_>,
    env: &mut MenuEnv<'_>,
    area: &Area,
    y: f32,
    title: &str,
    lit: bool,
) -> f32 {
    let s = area.s;
    shadowed(
        painter,
        env,
        title,
        area.x + 11.0 * s,
        y + 5.0 * s,
        TextStyle::new(10.0 * s).weight(Weight::Bold).caps(0.16),
        white(if lit { 0.72 } else { 0.46 }),
        s,
    );
    y + 24.0 * s
}

/// The amber disc with a "!" (a clash), centred on `(cx, cy)`.
pub(super) fn draw_warn_mark(painter: &mut Painter<'_>, cx: f32, cy: f32, s: f32, color: Rgba) {
    painter.circle(cx, cy, 5.0 * s, color);
    let ink = css(0, 0, 0, 0.85);
    painter.rect(
        Rect::new(cx - 0.65 * s, cy - 3.0 * s, 1.3 * s, 3.6 * s),
        ink,
    );
    painter.rect(
        Rect::new(cx - 0.65 * s, cy + 1.5 * s, 1.3 * s, 1.3 * s),
        ink,
    );
}

/// The Keybinds view's own hit targets. Must stay `Copy` and `PartialEq`: it lives inside the
/// shell's `Target`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum KeysTarget {
    /// The menu key's row: a click captures, a right click puts Right Shift back.
    MenuKey,
    /// Key setting `.1` of mod `.0`: a click captures, a right click puts its default back.
    Key(usize, usize),
}

/// The Keybinds view's state: window 1's badge, counted once per settings generation (the rows
/// themselves are recomputed while the view is open).
#[derive(Debug, Default)]
pub(super) struct KeysState {
    /// The settings generation and whether Minecraft's controls were known when `badge` was
    /// counted (generation 0: never).
    seen: (u64, bool),
    badge: NavBadge,
}

/// What one key row shows.
struct KeyRow<'a> {
    key: ModKey,
    label: &'a str,
    /// Grey text after the label (a chat macro's message).
    extra: Option<&'a str>,
    /// The "Also: ..." line and whether it clashes right now.
    note: Option<(String, bool)>,
    capturing: bool,
    default_name: String,
}

impl ModMenuScreen {
    /// Window 1's badge for Keybinds: how many keys mods use (the menu key included), and an
    /// amber mark while two keys that work right now share a key. Counted again only when the
    /// settings generation moves (no allocation either way).
    pub(super) fn keybinds_badge(&mut self, env: &MenuEnv<'_>) -> NavBadge {
        let settings = env.settings;
        let vanilla = env.game.vanilla_keys;
        let key = (settings.generation(), vanilla.is_some());
        if self.keys.seen == key {
            return self.keys.badge;
        }
        let mut count = 0;
        let mut warn = false;
        for key in mod_keys(settings) {
            count += 1;
            warn = warn || active_clash(settings, vanilla, key);
        }
        self.keys.seen = key;
        self.keys.badge = NavBadge {
            count: Some(count),
            warn,
        };
        self.keys.badge
    }

    /// Window 2 while the Keybinds view is open: the menu key, then every mod's keys.
    pub(super) fn draw_keybinds_list(
        &mut self,
        painter: &mut Painter<'_>,
        env: &mut MenuEnv<'_>,
        rect: Rect,
        s: f32,
    ) {
        let settings = env.settings;
        let vanilla = env.game.vanilla_keys;
        let all: Vec<ModKey> = mod_keys(settings).collect();
        let clashing = all
            .iter()
            .filter(|key| active_clash(settings, vanilla, **key))
            .count();
        let subtitle = match clashing {
            0 => format!("{} keys your mods use", all.len()),
            1 => "1 key shares its key with another".to_owned(),
            n => format!("{n} keys share a key with another"),
        };
        draw_title(painter, env, rect, "Keybinds", &subtitle, 0.0, s);
        let area = self.begin_body(painter, rect, Pane::List, s);
        let mut y = draw_wrapped(
            painter,
            env,
            "Click a key, then press the new one: Esc cancels, Backspace unbinds. Right-click \
             puts the default back. Keys of mods that are off are dimmed.",
            area.x + 4.0 * s,
            area.y,
            area.w - 8.0 * s,
            TextStyle::new(11.5 * s),
            white(0.62),
            s,
        ) + 8.0 * s;
        y = draw_section(painter, env, &area, y, "Menu", true);
        let menu = all[0];
        let (names, hot) = clash_names(settings, vanilla, &all, menu);
        let row = KeyRow {
            key: menu,
            label: "Open and close this menu",
            extra: None,
            note: (!names.is_empty()).then(|| (format!("Also: {}", names.join(", ")), hot)),
            capturing: self.menu_key_capture,
            default_name: key_name(mc_mods::DEFAULT_MENU_KEY),
        };
        y = self.draw_key_row(painter, env, &area, y, &row);
        let order: Vec<usize> = self.order.mods.iter().map(|&(m, _)| m).collect();
        for m in order {
            let keys: Vec<ModKey> = all
                .iter()
                .copied()
                .filter(|key| matches!(key.owner, Owner::Mod(km, _) if km == m))
                .collect();
            if keys.is_empty() {
                continue;
            }
            let descriptor = settings.catalog()[m];
            let on = settings.enabled_at(m);
            let title = if on {
                descriptor.name.to_owned()
            } else {
                format!("{}  \u{b7}  off", descriptor.name)
            };
            y = draw_section(painter, env, &area, y + 8.0 * s, &title, on);
            for key in keys {
                let Owner::Mod(_, o) = key.owner else {
                    continue;
                };
                let option = &descriptor.options[o];
                // A chat macro's key shows its message.
                let extra = menu_rules::PAIRED
                    .iter()
                    .filter(|paired| paired.mod_id == descriptor.id)
                    .flat_map(|paired| paired.pairs.iter())
                    .find(|(left, _)| *left == option.id)
                    .and_then(|(_, right)| descriptor.option_index(right))
                    .and_then(|t| match settings.value_at(m, t) {
                        Some(OptionValue::Text(text)) if !text.is_empty() => Some(text.as_ref()),
                        _ => None,
                    });
                let (names, hot) = clash_names(settings, vanilla, &all, key);
                let default_name = match option.default {
                    OptionValue::Key(code) => key_name(code),
                    _ => key_name(0),
                };
                let row = KeyRow {
                    key,
                    label: option.label,
                    extra,
                    note: (!names.is_empty()).then(|| (format!("Also: {}", names.join(", ")), hot)),
                    capturing: self.capture == Some((m, o)),
                    default_name,
                };
                y = self.draw_key_row(painter, env, &area, y, &row);
            }
        }
        let reveal = self.nav_reveal(Pane::List);
        self.end_body(painter, Pane::List, area, y, reveal);
    }

    /// One key row at `y`: the label (and a macro's message), the key's badge, and the "Also:
    /// ..." line when another key or control uses it. Returns the y below it.
    fn draw_key_row(
        &mut self,
        painter: &mut Painter<'_>,
        env: &mut MenuEnv<'_>,
        area: &Area,
        y: f32,
        row: &KeyRow<'_>,
    ) -> f32 {
        let s = area.s;
        let h = if row.note.is_some() {
            ROW_NOTE_H
        } else {
            ROW_H
        } * s;
        let rect = Rect::new(area.x, y, area.w, h);
        let line = ROW_H * s;
        let hit = area.hit(painter, rect);
        let hover =
            self.drag.is_none() && hit.is_some_and(|r| r.contains(self.cursor.0, self.cursor.1));
        let list = Pane::List.index();
        let index = self.nav[list].len();
        let focused = self.keyboard && self.pane == Pane::List && self.cursors[list] == Some(index);
        let radius = 11.0 * s;
        if row.note.as_ref().is_some_and(|(_, hot)| *hot) {
            painter.round_rect(rect, radius, Fill::Solid(css(255, 196, 82, 0.07)));
        }
        if hover || row.capturing {
            painter.round_rect(rect, radius, Fill::Solid(white(0.08)));
            painter.round_rect_stroke(rect, radius, 1.0, 0.5, white(0.13));
        }
        if focused {
            draw_focus_ring(painter, rect, radius, s);
        }
        let saved_alpha = painter.alpha;
        if !row.key.active {
            painter.alpha *= 0.5;
        }
        // The badge first: the label gets the room left of it.
        let text = if row.capturing {
            "Press a key\u{2026}".to_owned()
        } else {
            key_name(row.key.code)
        };
        let badge = draw_badge(
            painter,
            env,
            &text,
            rect.right() - 8.0 * s,
            y,
            line,
            s,
            hover || row.capturing,
            false,
        );
        if row.capturing {
            painter.round_rect_stroke(badge, 8.0 * s, 1.0, 0.5, white(0.85));
        }
        let style = TextStyle::new(12.5 * s);
        let (_, lh) = env.text.metrics(style);
        let label_x = rect.x + 12.0 * s;
        let room = (badge.x - 10.0 * s - label_x).max(1.0);
        let saved_clip = painter.clip();
        let clip = Rect::new(label_x - 2.0 * s, y, room + 2.0 * s, line);
        painter.set_clip(Some(saved_clip.map_or(clip, |c| c.intersect(&clip))));
        let advance = shadowed(
            painter,
            env,
            row.label,
            label_x,
            y + (line - lh) * 0.5,
            style,
            white(0.88),
            s,
        );
        if let Some(extra) = row.extra {
            let small = TextStyle::new(11.0 * s);
            let (_, slh) = env.text.metrics(small);
            shadowed(
                painter,
                env,
                extra,
                label_x + advance + 8.0 * s,
                y + (line - slh) * 0.5,
                small,
                white(0.5),
                s,
            );
        }
        painter.set_clip(saved_clip);
        if let Some((note, hot)) = &row.note {
            let small = TextStyle::new(11.0 * s);
            let (_, slh) = env.text.metrics(small);
            let ny = y + line - 4.0 * s;
            let color = if *hot { CLASH } else { white(0.5) };
            draw_warn_mark(painter, label_x + 5.0 * s, ny + slh * 0.5, s * 0.85, color);
            let clip = Rect::new(
                label_x,
                ny - 2.0 * s,
                rect.right() - label_x - 8.0 * s,
                slh + 4.0 * s,
            );
            painter.set_clip(Some(saved_clip.map_or(clip, |c| c.intersect(&clip))));
            shadowed(
                painter,
                env,
                note,
                label_x + 15.0 * s,
                ny,
                small,
                if *hot { CLASH } else { white(0.55) },
                s,
            );
            painter.set_clip(saved_clip);
        }
        painter.alpha = saved_alpha;
        let target = match row.key.owner {
            Owner::Menu => KeysTarget::MenuKey,
            Owner::Mod(m, o) => KeysTarget::Key(m, o),
        };
        if let Some(r) = area.hit(painter, badge.inset(-3.0 * s)) {
            // The shell's own targets under ours: `menu_key_rect` and tests find them.
            let shell = match row.key.owner {
                Owner::Menu => Target::MenuKey,
                Owner::Mod(m, o) => Target::Key(m, o),
            };
            self.hits.push((r, shell));
        }
        if let Some(r) = hit {
            self.hits.push((r, Target::Keys(target)));
            if r.contains(self.cursor.0, self.cursor.1) && !row.capturing {
                let tip = format!(
                    "Click, then press a key. Right-click puts {} back.",
                    row.default_name
                );
                self.set_tooltip(&tip, self.cursor);
            }
        }
        self.nav[list].push(NavRow {
            enter: Target::Keys(target),
            space: None,
            span: area.span(y, y + h),
        });
        y + h + 2.0 * s
    }

    /// Window 3 while the Keybinds view is open: Minecraft's controls, read-only, with the ones
    /// a mod key also uses marked. Always draws the window.
    pub(super) fn draw_keybinds_details(
        &mut self,
        painter: &mut Painter<'_>,
        env: &mut MenuEnv<'_>,
        rect: Rect,
        s: f32,
    ) -> bool {
        draw_title(
            painter,
            env,
            rect,
            "Minecraft controls",
            "Change them in Options \u{203a} Controls",
            0.0,
            s,
        );
        let area = self.begin_body(painter, rect, Pane::Details, s);
        let settings = env.settings;
        let Some(vanilla) = env.game.vanilla_keys else {
            let y = draw_wrapped(
                painter,
                env,
                "Minecraft's own controls show here, so you can see which keys are free.",
                area.x + 4.0 * s,
                area.y,
                area.w - 8.0 * s,
                TextStyle::new(11.5 * s),
                white(0.62),
                s,
            );
            self.end_body(painter, Pane::Details, area, y, None);
            return true;
        };
        let all: Vec<ModKey> = mod_keys(settings).filter(|key| key.code != 0).collect();
        let mut y = draw_wrapped(
            painter,
            env,
            "Your controls from options.txt. A key a mod also uses is marked: pick another key \
             for the mod, or change the control in Options \u{203a} Controls.",
            area.x + 4.0 * s,
            area.y,
            area.w - 8.0 * s,
            TextStyle::new(11.5 * s),
            white(0.62),
            s,
        ) + 8.0 * s;
        // In the order of the Controls screen (vanilla's `KeyBinding.compareTo`): by category
        // name, then by control name.
        let mut controls: Vec<(&str, String, i32)> = vanilla
            .iter()
            .map(|b| {
                (
                    category_name(b.category()),
                    control_name(b.description()),
                    b.binding().to_lwjgl(),
                )
            })
            .collect();
        controls.sort_by(|a, b| a.0.cmp(b.0).then_with(|| a.1.cmp(&b.1)));
        let mut category: Option<&str> = None;
        for (cat, name, code) in &controls {
            if category != Some(*cat) {
                if category.is_some() {
                    y += 6.0 * s;
                }
                category = Some(cat);
                y = draw_section(painter, env, &area, y, cat, true);
            }
            y = self.draw_control_row(painter, env, &area, y, name, *code, &all);
        }
        y = draw_section(painter, env, &area, y + 6.0 * s, "Fixed keys", true);
        for (code, name) in FIXED_KEYS {
            y = self.draw_control_row(painter, env, &area, y, name, code, &all);
        }
        self.end_body(painter, Pane::Details, area, y, None);
        true
    }

    /// One read-only control row: its name and key, and "Also: ..." when a mod key uses it.
    #[allow(clippy::too_many_arguments)]
    fn draw_control_row(
        &mut self,
        painter: &mut Painter<'_>,
        env: &mut MenuEnv<'_>,
        area: &Area,
        y: f32,
        name: &str,
        code: i32,
        all: &[ModKey],
    ) -> f32 {
        let s = area.s;
        let settings = env.settings;
        let users: Vec<&ModKey> = all
            .iter()
            .filter(|key| code != 0 && key.code == code)
            .collect();
        let hot = users.iter().any(|key| key.active);
        let line = 30.0 * s;
        let h = if users.is_empty() { line } else { 44.0 * s };
        let rect = Rect::new(area.x, y, area.w, h);
        if !users.is_empty() {
            let fill = if hot {
                css(255, 196, 82, 0.08)
            } else {
                white(0.04)
            };
            painter.round_rect(rect, 10.0 * s, Fill::Solid(fill));
        }
        let badge = draw_badge(
            painter,
            env,
            &key_name(code),
            rect.right() - 8.0 * s,
            y,
            line,
            s,
            false,
            false,
        );
        let style = TextStyle::new(12.0 * s);
        let (_, lh) = env.text.metrics(style);
        let label_x = rect.x + 12.0 * s;
        let saved_clip = painter.clip();
        let clip = Rect::new(label_x - 2.0 * s, y, badge.x - label_x - 6.0 * s, line);
        painter.set_clip(Some(saved_clip.map_or(clip, |c| c.intersect(&clip))));
        shadowed(
            painter,
            env,
            name,
            label_x,
            y + (line - lh) * 0.5,
            style,
            white(if code == 0 { 0.5 } else { 0.84 }),
            s,
        );
        painter.set_clip(saved_clip);
        if !users.is_empty() {
            let names: Vec<String> = users
                .iter()
                .map(|key| owner_name(settings, key.owner))
                .collect();
            let note = format!("Also: {}", names.join(", "));
            let small = TextStyle::new(11.0 * s);
            let (_, slh) = env.text.metrics(small);
            let ny = y + line - 3.0 * s;
            let color = if hot { CLASH } else { white(0.55) };
            draw_warn_mark(painter, label_x + 5.0 * s, ny + slh * 0.5, s * 0.85, color);
            let clip = Rect::new(
                label_x,
                ny - 2.0 * s,
                rect.right() - label_x - 8.0 * s,
                slh + 4.0 * s,
            );
            painter.set_clip(Some(saved_clip.map_or(clip, |c| c.intersect(&clip))));
            shadowed(painter, env, &note, label_x + 15.0 * s, ny, small, color, s);
            painter.set_clip(saved_clip);
        }
        if let Some(r) = area.hit(painter, rect) {
            if r.contains(self.cursor.0, self.cursor.1) {
                self.set_tooltip(
                    "Minecraft's control: change it in Options \u{203a} Controls.",
                    self.cursor,
                );
            }
        }
        y + h + 2.0 * s
    }

    /// A click on one of the view's targets (`button`: 0 left, 1 right).
    pub(super) fn activate_keys(
        &mut self,
        target: KeysTarget,
        button: u8,
        input: &mut MenuInput<'_>,
    ) {
        match (target, button) {
            (KeysTarget::MenuKey, 0) => {
                self.capture = None;
                self.menu_key_capture = true;
            }
            (KeysTarget::MenuKey, 1) => {
                input.settings.set_menu_key(mc_mods::DEFAULT_MENU_KEY);
            }
            (KeysTarget::Key(m, o), 0) => {
                self.menu_key_capture = false;
                self.capture = Some((m, o));
            }
            (KeysTarget::Key(m, o), 1) => {
                let default = input
                    .settings
                    .catalog()
                    .get(m)
                    .and_then(|descriptor| descriptor.options.get(o))
                    .map(|option| option.default.clone());
                if let Some(default) = default {
                    input.settings.set_value_at(m, o, default);
                }
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vanilla_names_match_en_us_lang() {
        assert_eq!(control_name("key.drop"), "Drop Item");
        assert_eq!(control_name("key.playerlist"), "List Players");
        assert_eq!(control_name("key.hotbar.8"), "Hotbar Slot 8");
        assert_eq!(control_name("key.unknown"), "key.unknown");
        assert_eq!(category_name("key.categories.misc"), "Miscellaneous");
        // Every vanilla control has a real name.
        for binding in KeyBindings::vanilla().iter() {
            assert!(
                !control_name(binding.description()).starts_with("key."),
                "{}",
                binding.description()
            );
            assert!(!category_name(binding.category()).starts_with("key."));
        }
    }

    /// The clash check: a mod key on a vanilla control's key, two mod keys on one key, the
    /// menu key; only keys that work right now (their mod on) make the warning amber.
    #[test]
    fn clashes_are_found_and_only_active_ones_warn() {
        let mut settings = ModSettings::new();
        let vanilla = KeyBindings::vanilla();
        let zoom = settings.index_of("zoom").unwrap();
        let key = settings.catalog()[zoom].option_index("key").unwrap();
        // Zoom (off by default) on Q, Minecraft's Drop Item.
        settings.set_value_at(zoom, key, OptionValue::Key(16));
        let all: Vec<ModKey> = mod_keys(&settings).collect();
        let zk = *all
            .iter()
            .find(|k| k.owner == Owner::Mod(zoom, key))
            .unwrap();
        assert!(!zk.active, "zoom is off");
        let (names, hot) = clash_names(&settings, Some(&vanilla), &all, zk);
        assert_eq!(names, ["Drop Item"]);
        assert!(!hot, "a key of a mod that is off does nothing yet");
        assert!(!active_clash(&settings, Some(&vanilla), zk));
        settings.set_enabled_at(zoom, true);
        let all: Vec<ModKey> = mod_keys(&settings).collect();
        let zk = *all
            .iter()
            .find(|k| k.owner == Owner::Mod(zoom, key))
            .unwrap();
        assert!(active_clash(&settings, Some(&vanilla), zk));
        let (_, hot) = clash_names(&settings, Some(&vanilla), &all, zk);
        assert!(hot);
        // Without the vanilla list (tests, harnesses) only mod keys count.
        assert!(!active_clash(&settings, None, zk));
        // The menu key on the same key as zoom: both see each other.
        settings.set_menu_key(16);
        let all: Vec<ModKey> = mod_keys(&settings).collect();
        let (names, hot) = clash_names(&settings, None, &all, all[0]);
        assert_eq!(names, ["Zoom key (Zoom)"]);
        assert!(hot);
        // An unbound key never clashes.
        settings.set_value_at(zoom, key, OptionValue::Key(0));
        settings.set_menu_key(mc_mods::DEFAULT_MENU_KEY);
        let all: Vec<ModKey> = mod_keys(&settings).collect();
        assert!(all
            .iter()
            .all(|k| !active_clash(&settings, Some(&vanilla), *k)));
        // F3 is vanilla's own.
        settings.set_value_at(zoom, key, OptionValue::Key(61));
        let all: Vec<ModKey> = mod_keys(&settings).collect();
        let zk = *all
            .iter()
            .find(|k| k.owner == Owner::Mod(zoom, key))
            .unwrap();
        let (names, _) = clash_names(&settings, Some(&vanilla), &all, zk);
        assert_eq!(names, ["Debug screen (F3)"]);
        // So is F4, the in-game shader toggle.
        settings.set_value_at(zoom, key, OptionValue::Key(62));
        let all: Vec<ModKey> = mod_keys(&settings).collect();
        let zk = *all
            .iter()
            .find(|k| k.owner == Owner::Mod(zoom, key))
            .unwrap();
        let (names, hot) = clash_names(&settings, Some(&vanilla), &all, zk);
        assert_eq!(names, ["Shader on/off (F4)"]);
        assert!(hot && active_clash(&settings, Some(&vanilla), zk));
    }

    /// The fresh install's keys clash with nothing (the defaults were picked around vanilla's).
    #[test]
    fn a_fresh_install_has_no_clash() {
        let mut settings = ModSettings::new();
        let vanilla = KeyBindings::vanilla();
        for m in 0..settings.catalog().len() {
            settings.set_enabled_at(m, true);
        }
        for key in mod_keys(&settings) {
            assert!(
                !active_clash(&settings, Some(&vanilla), key),
                "{}",
                owner_name(&settings, key.owner)
            );
        }
        let count = mod_keys(&settings).count();
        assert!(
            count >= 15,
            "the menu key, zoom, waypoints, markers, macros, spotify..."
        );
    }
}
