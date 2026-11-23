//! The HUD editor: drag to move, scroll to resize, snapping, nudging, and the settings cog.

use std::time::Instant;

use mc_mods::HudRect;

use super::extras::ExtrasTarget;
use super::paint::{black, css, white, Fill, Painter, Rect, Rgba};
use super::state::{Drag, Mode, NavRow, Pane, Target};
use super::text::{TextStyle, Weight};
use super::widgets::{draw_pill, ms_since, PILL_H};
use super::{MenuEnv, MenuInput, ModMenuScreen};
use crate::modhud::ModHud;

/// The HUD editor's light dim (research: 25 %), drawn under the mod HUD by the client.
pub const EDITOR_DIM_ARGB: u32 = 0x4000_0000;
/// How long a "Reset all" first click stays armed.
pub(super) const RESET_ALL_ARM_MS: u128 = 3000;
/// Snap distance in GUI pixels (research: 4).
pub(super) const SNAP_GUI_PX: f32 = 4.0;
/// How far, in GUI pixels, the pointer may stray from the HUD element the wheel is resizing, or
/// from where it last worked on it, before the wheel lets go.
pub(super) const GRIP_GUI_PX: f32 = 6.0;
/// How long the element the wheel just resized keeps its hover outline and scale label.
pub(super) const WHEEL_HOLD_MS: u128 = 1000;
/// Accent for snap guides (a red that stays readable on sky and terrain).
pub(super) const GUIDE: Rgba = css(255, 90, 90, 0.95);

impl ModMenuScreen {
    // ---- HUD editor ------------------------------------------------------------------

    pub(super) fn enter_editor(&mut self, now: Instant) {
        self.mode = Mode::HudEditor;
        self.search_selected = false;
        self.return_to_hud = false;
        self.mode_since = now;
        self.color_editor = None;
        self.capture = None;
        self.text_focus = None;
        self.name_focus = None;
        self.menu_key_capture = false;
        self.drag = None;
        self.hud_grip = None;
        // The tray starts closed each time.
        self.extras.tray_open = false;
    }

    pub(super) fn leave_editor(&mut self, now: Instant) {
        self.mode = Mode::Menu;
        self.return_to_hud = false;
        self.mode_since = now;
        // The windows play their opening animation again.
        self.opened_at = now;
        self.drag = None;
        self.hud_grip = None;
        self.guides = mc_mods::SnapGuides::default();
        self.reset_all_armed = None;
    }

    pub(super) fn reset_all(&mut self, input: &mut MenuInput<'_>) {
        let armed = self.reset_all_armed.is_some_and(|at| {
            input.now.saturating_duration_since(at).as_millis() < RESET_ALL_ARM_MS
        });
        if armed {
            let targets: Vec<(usize, usize)> = input
                .hud
                .slots()
                .iter()
                .map(|slot| (slot.mod_index, slot.element_index))
                .collect();
            for (m, e) in targets {
                input.settings.reset_placement_at(m, e);
            }
            self.reset_all_armed = None;
        } else {
            self.reset_all_armed = Some(input.now);
        }
    }

    pub(super) fn editor_pressed(&mut self, x: f32, y: f32, button: u8, input: &mut MenuInput<'_>) {
        // The tray's buttons and panel take clicks too, so a
        // click on them never starts a drag.
        if let Some(
            target @ (Target::Done | Target::ResetAll | Target::HudSettings(_) | Target::Extras(_)),
        ) = self.hit(x, y)
        {
            if button == 0 {
                self.pressed = Some((target, button));
            }
            return;
        }
        let (gx, gy) = (x / input.gui_scale, y / input.gui_scale);
        let Some(slot) = input.hud.hit(gx, gy) else {
            // A click on empty space lets go of the gripped element.
            self.hud_grip = None;
            return;
        };
        let Some(info) = input.hud.slots().get(slot) else {
            return;
        };
        let (m, e) = (info.mod_index, info.element_index);
        self.hud_last = Some(slot);
        self.hud_grip = Some((slot, (gx, gy)));
        match button {
            0 => {
                if let Some(rect) = info.rect {
                    self.drag = Some(Drag::Hud {
                        slot,
                        start: (gx, gy),
                        grab: (gx - rect.x, gy - rect.y),
                        size: (rect.w, rect.h),
                        moving: false,
                        placement: None,
                    });
                }
            }
            1 => {
                input.settings.reset_placement_at(m, e);
            }
            _ => {}
        }
    }

    pub(super) fn editor_dragged(&mut self, x: f32, y: f32, input: &mut MenuInput<'_>) {
        let Some(Drag::Hud {
            slot,
            start,
            grab,
            size,
            moving,
            placement,
        }) = self.drag
        else {
            return;
        };
        let (gx, gy) = (x / input.gui_scale, y / input.gui_scale);
        let moving = moving || (gx - start.0).abs() >= 2.0 || (gy - start.1).abs() >= 2.0;
        if !moving {
            return;
        }
        let screen = input.gui_screen;
        let mut rect = HudRect::new(
            (gx - grab.0).clamp(0.0, (screen.0 - size.0).max(0.0)),
            (gy - grab.1).clamp(0.0, (screen.1 - size.1).max(0.0)),
            size.0,
            size.1,
        );
        self.guides = mc_mods::SnapGuides::default();
        if !input.alt {
            let others: Vec<HudRect> = input
                .hud
                .slots()
                .iter()
                .enumerate()
                .filter(|(i, s)| *i != slot && s.enabled)
                .filter_map(|(_, s)| s.rect)
                .collect();
            let (snapped, guides) = mc_mods::snap_rect_with(rect, screen, SNAP_GUI_PX, &others);
            rect = snapped;
            self.guides = guides;
        }
        let scale = input
            .hud
            .slots()
            .get(slot)
            .map_or(1.0, |s| s.placement.scale);
        let new_placement = input
            .hud
            .placement_for(slot, &rect, scale, screen)
            .or(placement);
        if let Some(p) = new_placement {
            input.hud.preview_placement(slot, p);
        }
        self.drag = Some(Drag::Hud {
            slot,
            start,
            grab,
            size,
            moving,
            placement: new_placement,
        });
    }

    pub(super) fn editor_released(&mut self, button: u8, input: &mut MenuInput<'_>) {
        if let Some((target, pressed)) = self.pressed.take() {
            if pressed == button && self.hit(self.cursor.0, self.cursor.1) == Some(target) {
                self.activate(target, button, input);
            }
        }
        if button != 0 {
            return;
        }
        if let Some(Drag::Hud {
            slot,
            placement: Some(placement),
            moving: true,
            ..
        }) = self.drag
        {
            if let Some(info) = input.hud.slots().get(slot) {
                let (m, e) = (info.mod_index, info.element_index);
                input.settings.set_placement_at(m, e, placement);
            }
        }
        self.drag = None;
        self.guides = mc_mods::SnapGuides::default();
    }

    pub(super) fn nudge(&mut self, dx: f32, dy: f32, input: &mut MenuInput<'_>) {
        let Some(slot) = self.hud_hover.or(self.hud_last) else {
            return;
        };
        let Some(info) = input.hud.slots().get(slot) else {
            return;
        };
        let (m, e, scale) = (info.mod_index, info.element_index, info.placement.scale);
        let Some(rect) = info.rect else {
            return;
        };
        let screen = input.gui_screen;
        let moved = HudRect::new(
            (rect.x + dx).clamp(0.0, (screen.0 - rect.w).max(0.0)),
            (rect.y + dy).clamp(0.0, (screen.1 - rect.h).max(0.0)),
            rect.w,
            rect.h,
        );
        if let Some(p) = input.hud.placement_for(slot, &moved, scale, screen) {
            input.settings.set_placement_at(m, e, p);
        }
    }

    /// The HUD element the wheel resizes with the pointer at the GUI point `(gx, gy)`: the one
    /// under the pointer, or else the gripped one (`hud_grip`: the last element hovered,
    /// pressed or scrolled) while the pointer is within [`GRIP_GUI_PX`] of its rectangle or of
    /// the point where it last worked on it.
    ///
    /// Scaling about the anchor shrinks an element towards its pinned point, so it can leave a
    /// pointer that stays still; the grip keeps the wheel on it, so scrolling back up grows it
    /// under the pointer again.
    pub(super) fn scroll_target(&self, gx: f32, gy: f32, hud: &ModHud) -> Option<usize> {
        if let Some(slot) = hud.hit(gx, gy) {
            return Some(slot);
        }
        let (slot, (ax, ay)) = self.hud_grip?;
        let rect = hud.slots().get(slot).filter(|s| s.enabled)?.rect?;
        let near_point = (gx - ax).abs() <= GRIP_GUI_PX && (gy - ay).abs() <= GRIP_GUI_PX;
        let near_rect = gx >= rect.x - GRIP_GUI_PX
            && gx <= rect.right() + GRIP_GUI_PX
            && gy >= rect.y - GRIP_GUI_PX
            && gy <= rect.bottom() + GRIP_GUI_PX;
        (near_point || near_rect).then_some(slot)
    }

    /// Whether the wheel resized `slot` within the last [`WHEEL_HOLD_MS`] and still grips it:
    /// it then keeps its hover outline and scale label, also once it has shrunk off the pointer.
    pub(super) fn wheel_holds(&self, slot: usize, now: Instant) -> bool {
        self.hud_grip.is_some_and(|(gripped, _)| gripped == slot)
            && self
                .hud_wheel_at
                .is_some_and(|at| now.saturating_duration_since(at).as_millis() < WHEEL_HOLD_MS)
    }

    pub(super) fn draw_editor(&mut self, painter: &mut Painter<'_>, env: &mut MenuEnv<'_>, s: f32) {
        let g = env.gui_scale;
        let label_style = TextStyle::new(11.0 * s).weight(Weight::Semibold);
        let dragging = match self.drag {
            Some(Drag::Hud {
                slot, moving: true, ..
            }) => Some(slot),
            _ => None,
        };
        for (i, slot) in env.hud.slots().iter().enumerate() {
            let Some(rect) = slot.rect.filter(|_| slot.enabled) else {
                continue;
            };
            let r = Rect::new(rect.x * g, rect.y * g, rect.w * g, rect.h * g);
            // The element "Move on screen" or the tray just
            // picked glows for a moment, so the eye finds it.
            let picked = self
                .extras
                .picked
                .filter(|&(p, at)| p == i && ms_since(at, env.now) < PICK_MS)
                .map(|(_, at)| ms_since(at, env.now) / PICK_MS);
            if let Some(t) = picked {
                let glow = (1.0 - t).clamp(0.0, 1.0);
                painter.round_rect_shadow(
                    r.inset(-2.0 * s),
                    3.0 * s,
                    [10.0 * s; 4],
                    white(0.35 * glow),
                );
                painter.rect(r, white(0.12 * glow));
            }
            // The element the wheel is resizing looks hovered, so its
            // outline and scale label stay up while it shrinks out from under the pointer.
            let hovered =
                self.hud_hover == Some(i) || self.wheel_holds(i, env.now) || picked.is_some();
            let active = dragging == Some(i);
            if active {
                painter.rect(r, white(0.10));
            }
            let alpha = if active {
                1.0
            } else if hovered {
                0.9
            } else {
                0.35
            };
            let px = 1.0_f32.max(s.round());
            // A 1-physical-pixel outline just outside the element.
            let o = r.inset(-px);
            painter.rect(Rect::new(o.x, o.y, o.w, px), white(alpha));
            painter.rect(Rect::new(o.x, o.bottom() - px, o.w, px), white(alpha));
            painter.rect(Rect::new(o.x, o.y + px, px, o.h - 2.0 * px), white(alpha));
            painter.rect(
                Rect::new(o.right() - px, o.y + px, px, o.h - 2.0 * px),
                white(alpha),
            );
            // Keep the settings button outside the drag area, and inside the window even
            // for elements anchored to an edge. Its hit target takes precedence over drags.
            let gear_size = 20.0 * s;
            let gear_x = if o.right() + gear_size + 4.0 * s <= env.screen.0 {
                o.right() + 4.0 * s
            } else {
                (o.x - gear_size - 4.0 * s).max(0.0)
            };
            let gear = Rect::new(
                gear_x,
                o.y.clamp(0.0, (env.screen.1 - gear_size).max(0.0)),
                gear_size,
                gear_size,
            );
            let gear_hovered = gear.contains(self.cursor.0, self.cursor.1);
            painter.round_rect(
                gear,
                5.0 * s,
                Fill::Solid(black(if gear_hovered { 0.85 } else { 0.60 })),
            );
            draw_settings_gear(painter, gear, s, if gear_hovered { 1.0 } else { 0.75 });
            self.hits.push((gear, Target::HudSettings(slot.mod_index)));
            if hovered || active {
                let text = format!("{}  {:.1}x", slot.name, slot.placement.scale);
                let tw = env.text.measure(&text, label_style, env.mc_font);
                let (_, lh) = env.text.metrics(label_style);
                let above = o.y - lh - 8.0 * s;
                let ty = if above > 2.0 {
                    above
                } else {
                    o.bottom() + 4.0 * s
                };
                let tag = Rect::new(o.x, ty, tw + 12.0 * s, lh + 4.0 * s);
                painter.round_rect(tag, 6.0 * s, Fill::Solid(black(0.55)));
                env.text.draw(
                    painter,
                    env.mc_font,
                    &text,
                    tag.x + 6.0 * s,
                    tag.y + 2.0 * s,
                    label_style,
                    white(0.95),
                );
            }
        }
        // Snap guides.
        if dragging.is_some() {
            if let Some(x) = self.guides.vertical {
                let px = (x * g).round().clamp(0.0, env.screen.0 - 1.0);
                painter.rect(Rect::new(px, 0.0, 1.0, env.screen.1), GUIDE);
            }
            if let Some(y) = self.guides.horizontal {
                let py = (y * g).round().clamp(0.0, env.screen.1 - 1.0);
                painter.rect(Rect::new(0.0, py, env.screen.0, 1.0), GUIDE);
            }
        }
        // Hint bar and the Done / Reset all pills, bottom centre.
        let hint = "Drag to move  ·  Scroll to resize  ·  Right-click to reset  ·  Cog: settings  ·  Alt: free move  ·  Arrows nudge  ·  Esc when done";
        let hint_style = TextStyle::new(11.5 * s);
        let hw = env.text.measure(hint, hint_style, env.mc_font);
        let (_, hl) = env.text.metrics(hint_style);
        let pill_h = PILL_H * s;
        let done_w = 96.0 * s;
        let reset_w = 120.0 * s;
        let base_y = env.screen.1 - pill_h - 64.0 * s;
        let bar = Rect::new(
            (env.screen.0 - hw) * 0.5 - 14.0 * s,
            base_y - hl - 22.0 * s,
            hw + 28.0 * s,
            hl + 12.0 * s,
        );
        painter.round_rect(bar, bar.h * 0.5, Fill::Solid(black(0.45)));
        painter.round_rect_stroke(bar, bar.h * 0.5, 1.0, 0.5, white(0.18));
        env.text.draw(
            painter,
            env.mc_font,
            hint,
            bar.x + 14.0 * s,
            bar.y + 6.0 * s,
            hint_style,
            white(0.85),
        );
        // "Add to HUD" opens the tray of HUD mods that are off.
        let off = tray_mods(env);
        let add_label = format!("Add to HUD ({})", off.len());
        let add_w = env.text.measure(
            &add_label,
            TextStyle::new(12.5 * s).weight(Weight::Semibold),
            env.mc_font,
        ) + 40.0 * s;
        let total = done_w + 10.0 * s + reset_w + 10.0 * s + add_w;
        let done = Rect::new((env.screen.0 - total) * 0.5, base_y, done_w, pill_h);
        let reset = Rect::new(done.right() + 10.0 * s, base_y, reset_w, pill_h);
        let add = Rect::new(reset.right() + 10.0 * s, base_y, add_w, pill_h);
        let armed = self
            .reset_all_armed
            .is_some_and(|at| env.now.saturating_duration_since(at).as_millis() < RESET_ALL_ARM_MS);
        draw_pill(
            painter,
            env,
            done,
            "Done",
            done.contains(self.cursor.0, self.cursor.1),
            s,
        );
        draw_pill(
            painter,
            env,
            reset,
            if armed { "Click again" } else { "Reset all" },
            reset.contains(self.cursor.0, self.cursor.1) || armed,
            s,
        );
        draw_pill(
            painter,
            env,
            add,
            &add_label,
            add.contains(self.cursor.0, self.cursor.1) || self.extras.tray_open,
            s,
        );
        self.hits.push((done, Target::Done));
        self.hits.push((reset, Target::ResetAll));
        self.hits.push((add, Target::Extras(ExtrasTarget::Tray)));
        if self.extras.tray_open {
            self.draw_tray(painter, env, &off, add, bar.y, s);
        }
    }

    /// The Edit HUD tray: the HUD mods that are off, as chips in a panel above the
    /// pills; a click turns one on, and it shows up (glowing) to be placed.
    fn draw_tray(
        &mut self,
        painter: &mut Painter<'_>,
        env: &mut MenuEnv<'_>,
        off: &[usize],
        anchor: Rect,
        bottom: f32,
        s: f32,
    ) {
        let catalog = env.settings.catalog();
        let chip_style = TextStyle::new(12.0 * s).weight(Weight::Semibold);
        let (_, lh) = env.text.metrics(chip_style);
        let chip_h = lh + 12.0 * s;
        let pad = 14.0 * s;
        let max_w = (env.screen.0 - 40.0 * s).min(620.0 * s);
        // Lay the chips out in rows first.
        let mut chips: Vec<(usize, f32)> = Vec::with_capacity(off.len());
        for &m in off {
            let w = env.text.measure(catalog[m].name, chip_style, env.mc_font) + 34.0 * s;
            chips.push((m, w));
        }
        let mut rows: Vec<Vec<(usize, f32)>> = vec![Vec::new()];
        let mut row_w = 0.0;
        for chip in chips {
            if row_w + chip.1 > max_w - 2.0 * pad && !rows.last().is_some_and(Vec::is_empty) {
                rows.push(Vec::new());
                row_w = 0.0;
            }
            row_w += chip.1 + 8.0 * s;
            if let Some(row) = rows.last_mut() {
                row.push(chip);
            }
        }
        let widest = rows
            .iter()
            .map(|row| row.iter().map(|(_, w)| w + 8.0 * s).sum::<f32>())
            .fold(0.0_f32, f32::max);
        let title_style = TextStyle::new(12.5 * s).weight(Weight::Semibold);
        let note_style = TextStyle::new(11.0 * s);
        let (_, th) = env.text.metrics(title_style);
        let (_, nh) = env.text.metrics(note_style);
        let note = if off.is_empty() {
            "Every HUD mod is on already."
        } else {
            "Click one to turn it on: it appears on the HUD, ready to drag into place."
        };
        let note_w = env.text.measure(note, note_style, env.mc_font);
        let w = (widest + 2.0 * pad - 8.0 * s)
            .max(note_w + 2.0 * pad)
            .min(max_w);
        let chips_h = if off.is_empty() {
            0.0
        } else {
            rows.len() as f32 * (chip_h + 8.0 * s)
        };
        let h = pad + th + 4.0 * s + nh + 10.0 * s + chips_h + pad - 8.0 * s;
        let x =
            (anchor.center().0 - w * 0.5).clamp(10.0 * s, (env.screen.0 - w - 10.0 * s).max(0.0));
        let panel = Rect::new(x, (bottom - h - 12.0 * s).max(8.0 * s), w, h);
        painter.round_rect_shadow(panel, 16.0 * s, [16.0 * s; 4], black(0.35));
        painter.round_rect(panel, 16.0 * s, Fill::Solid(black(0.82)));
        painter.round_rect_stroke(panel, 16.0 * s, 1.0, 0.5, white(0.28));
        // The panel swallows clicks: nothing under it starts a drag.
        self.hits
            .push((panel, Target::Extras(ExtrasTarget::TrayPanel)));
        let mut y = panel.y + pad;
        env.text.draw(
            painter,
            env.mc_font,
            "Add to the HUD",
            panel.x + pad,
            y,
            title_style,
            white(0.96),
        );
        y += th + 4.0 * s;
        env.text.draw(
            painter,
            env.mc_font,
            note,
            panel.x + pad,
            y,
            note_style,
            white(0.62),
        );
        y += nh + 10.0 * s;
        for row in rows {
            let mut cx = panel.x + pad;
            for (m, cw) in row {
                let chip = Rect::new(cx, y, cw, chip_h);
                let hover = chip.contains(self.cursor.0, self.cursor.1);
                painter.round_rect(
                    chip,
                    chip_h * 0.5,
                    Fill::Solid(white(if hover { 0.22 } else { 0.10 })),
                );
                painter.round_rect_stroke(
                    chip,
                    chip_h * 0.5,
                    1.0,
                    0.5,
                    white(if hover { 0.6 } else { 0.3 }),
                );
                // A plus, then the name.
                let (px, py) = (chip.x + 13.0 * s, chip.y + chip_h * 0.5);
                painter.line(px - 4.0 * s, py, px + 4.0 * s, py, 1.4 * s, white(0.9));
                painter.line(px, py - 4.0 * s, px, py + 4.0 * s, 1.4 * s, white(0.9));
                env.text.draw(
                    painter,
                    env.mc_font,
                    catalog[m].name,
                    chip.x + 24.0 * s,
                    chip.y + 6.0 * s,
                    chip_style,
                    white(0.95),
                );
                self.hits
                    .push((chip, Target::Extras(ExtrasTarget::TrayAdd(m))));
                cx += cw + 8.0 * s;
            }
            y += chip_h + 8.0 * s;
        }
    }
}

/// How long the element "Move on screen" or the tray picked glows.
pub(super) const PICK_MS: f32 = 1600.0;

/// The mods with a HUD element that are off, by name (the tray's chips).
fn tray_mods(env: &MenuEnv<'_>) -> Vec<usize> {
    let mut mods: Vec<usize> = Vec::new();
    for slot in env.hud.slots() {
        if !env.settings.enabled_at(slot.mod_index) && !mods.contains(&slot.mod_index) {
            mods.push(slot.mod_index);
        }
    }
    mods.sort_by_key(|&m| env.settings.catalog()[m].name);
    mods
}

pub(super) fn draw_settings_gear(painter: &mut Painter<'_>, rect: Rect, s: f32, alpha: f32) {
    let (cx, cy) = rect.center();
    let color = white(alpha);
    for tooth in 0..8 {
        let angle = tooth as f32 * std::f32::consts::FRAC_PI_4;
        let (sin, cos) = angle.sin_cos();
        painter.line(
            cx + cos * 4.0 * s,
            cy + sin * 4.0 * s,
            cx + cos * 6.0 * s,
            cy + sin * 6.0 * s,
            2.5 * s,
            color,
        );
    }
    painter.circle(cx, cy, 4.5 * s, color);
    painter.circle(cx, cy, 2.0 * s, black(0.9));
}

// ---- The links between window 3 and the editor -------------
impl ModMenuScreen {
    /// Window 3's "Move on screen" button for mod `m`, left edge at `(x, y)` in the header's
    /// button row (26 dip tall, `area` is window 3's body). Returns the x where the next button
    /// goes: `x` itself when the mod draws nothing on the HUD. Push `Target::MoveOnScreen(m)`
    /// for the click (greyed out while the mod is off or `self.no_world`).
    pub(super) fn draw_move_on_screen(
        &mut self,
        painter: &mut Painter<'_>,
        env: &mut MenuEnv<'_>,
        m: usize,
        area: &super::layout::Area,
        (x, y): (f32, f32),
    ) -> f32 {
        // Only mods with an element in the HUD pipeline can be moved.
        if !env.hud.slots().iter().any(|slot| slot.mod_index == m) {
            return x;
        }
        let s = area.s;
        let on = env.settings.enabled_at(m);
        let enabled = on && !self.no_world;
        let r = self.draw_button(
            painter,
            env,
            "Move on screen",
            (x, y),
            26.0 * s,
            enabled,
            false,
            None,
            area,
        );
        if let Some(hit) = area.hit(painter, r) {
            if enabled {
                self.hits.push((hit, Target::MoveOnScreen(m)));
                self.nav[Pane::Details.index()].push(NavRow {
                    enter: Target::MoveOnScreen(m),
                    space: None,
                    span: area.span(r.y, r.bottom()),
                });
                if hit.contains(self.cursor.0, self.cursor.1) {
                    self.set_tooltip(
                        "Opens Edit HUD with this element picked: drag it, scroll to resize, \
                         arrows nudge. Esc comes back here.",
                        self.cursor,
                    );
                }
            } else if hit.contains(self.cursor.0, self.cursor.1) {
                let tip = if self.no_world {
                    "Moving HUD elements needs a world. Join a server, then open the menu again."
                } else {
                    "Turn it on to place it."
                };
                self.set_tooltip(tip, self.cursor);
            }
        }
        r.right() + 8.0 * s
    }

    /// A click on window 3's "Move on screen": Edit HUD opens with the mod's element picked (it
    /// glows, the arrows nudge it, the wheel resizes it); Esc in the editor comes back to this
    /// mod's settings. The cog on an element does the opposite trip.
    pub(super) fn activate_move_on_screen(&mut self, m: usize, input: &mut MenuInput<'_>) {
        if self.no_world || !input.settings.enabled_at(m) {
            return;
        }
        let Some(slot) = input.hud.slots().iter().position(|s| s.mod_index == m) else {
            return;
        };
        self.enter_editor(input.now);
        self.pick_hud_slot(slot, input);
    }

    /// Make HUD slot `slot` the editor's current element: it glows for a moment, the arrows
    /// nudge it and the wheel resizes it.
    pub(super) fn pick_hud_slot(&mut self, slot: usize, input: &MenuInput<'_>) {
        self.hud_last = Some(slot);
        self.extras.picked = Some((slot, input.now));
        if let Some(rect) = input.hud.slots().get(slot).and_then(|s| s.rect) {
            self.hud_grip = Some((slot, (rect.x + rect.w * 0.5, rect.y + rect.h * 0.5)));
        }
    }
}
