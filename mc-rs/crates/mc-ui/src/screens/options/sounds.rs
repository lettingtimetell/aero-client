//! The sound options: one volume slider per category, with Master double width.
//!
//! java: `client/gui/GuiScreenOptionsSounds.java`
//!
//! Vanilla's `GuiScreenOptionsSounds.Button` is a slider that draws no track — it reuses
//! `GuiButton`'s art with `getHoverState` pinned to 0 — which is exactly what S15's
//! [`Slider`] already does, so the only thing this file adds is the layout and the `"<name>: <n>%"` label rule.

use mc_assets::Language;
use mc_render::settings::{GameSettings, SOUND_CATEGORIES};

use crate::font::FontRenderer;
use crate::screen_stack::ScreenCtx;
use crate::screens::common::{draw_world_background, TITLE_COLOR};
use crate::screens::Action;
use crate::widget::draw::DrawList;
use crate::widget::{draw_centered_string, Button, Slider};

/// Side effects that the sound-options screen cannot perform itself.
///
/// The UI crate deliberately has no audio, filesystem, or network dependencies.  The caller
/// must drain these after every input callback and perform them in order.  This mirrors
/// `GuiScreenOptionsSounds.Button`: changing a slider applies the live engine volume first and
/// then calls `GameSettings.saveOptions`; releasing a grabbed slider plays one button click.
#[derive(Debug, Clone, PartialEq)]
pub enum SoundOptionsEvent {
    /// Apply this category's new level to the live sound engine.
    ApplyVolume {
        /// Vanilla sound-category id (`master`, `music`, ...).
        category: String,
        /// New normalised volume in `0..=1`.
        level: f32,
    },
    /// Persist `options.txt` and send the current client settings to the server, if connected.
    SaveSettings,
    /// Play `gui.button.press` once, at the end of a slider drag.
    PlayReleaseClick,
}

/// The sound options screen.
#[derive(Debug)]
pub struct SoundOptionsScreen {
    /// One slider per category, in `SOUND_CATEGORIES` order.
    pub sliders: Vec<(String, Slider)>,
    /// Whether a world is loaded.
    pub in_world: bool,
    done: Button,
    events: Vec<SoundOptionsEvent>,
    /// The slider pressed last: the arrow keys move it when none is hovered (X1, item 33).
    last_slider: Option<usize>,
}

/// A volume slider's label resolution: whole percent.
const VOLUME_STEPS: f32 = 100.0;

/// A volume position snapped to the whole percent its label shows (X1, item 33), so "Music: 37%" stores 0.37 (as the float its truncating label reads as 37).
fn snap_level(position: f32) -> f32 {
    crate::widget::snap_position(position, VOLUME_STEPS)
}

/// The label a volume slider shows: `off` at zero, otherwise a truncated percentage.
// java: GuiScreenOptionsSounds.java:78 (getSoundVolume)
#[must_use]
pub fn volume_label(name: &str, level: f32, lang: &Language) -> String {
    let value = if level == 0.0 {
        lang.tr("options.off").to_owned()
    } else {
        format!("{}%", (level * 100.0) as i32)
    };
    format!("{}: {value}", lang.tr(&format!("soundCategory.{name}")))
}

impl Default for SoundOptionsScreen {
    fn default() -> Self {
        Self::new(false)
    }
}

impl SoundOptionsScreen {
    /// A sound options screen.
    #[must_use]
    pub fn new(in_world: bool) -> Self {
        Self {
            sliders: Vec::new(),
            in_world,
            done: Button::new(200, 0, 0, "Done"),
            events: Vec::new(),
            last_slider: None,
        }
    }

    /// Drain side effects produced by the most recent input callbacks.
    ///
    /// Callers should preserve event order.  In particular, [`SoundOptionsEvent::ApplyVolume`]
    /// precedes [`SoundOptionsEvent::SaveSettings`] for every press and every drag frame.
    pub fn take_events(&mut self) -> Vec<SoundOptionsEvent> {
        std::mem::take(&mut self.events)
    }

    /// Lay out.
    ///
    /// Master occupies slots 0 *and* 1 (vanilla's `i = i + 2`) and is 310 wide, so the other
    /// eight categories start on the second row.
    // java: GuiScreenOptionsSounds.java:32
    pub fn init(&mut self, ctx: &ScreenCtx, settings: &GameSettings, lang: &Language) {
        let (w, h) = (ctx.width, ctx.height);
        // `Client::build_ui` re-runs this before every draw, but vanilla re-runs `initGui` only
        // on open and resize. Rebuilding the grabbed slider cleared its `field_146155_p` drag
        // latch one frame after the press, so a volume could be clicked but never dragged.
        let drag = self.sliders.iter().position(|(_, s)| s.dragging).map(|i| {
            (
                i,
                self.sliders[i].1.button.rect,
                self.sliders[i].1.position(),
            )
        });
        self.sliders.clear();
        let mut slot = 0i32;
        let place = |slot: i32| {
            (
                w / 2 - 155 + (slot % 2) * 160,
                h / 6 - 12 + 24 * (slot >> 1),
            )
        };
        for (i, name) in SOUND_CATEGORIES.iter().enumerate() {
            let master = i == 0;
            let (x, y) = place(slot);
            let level = settings.sound_level(name);
            let mut s = Slider::new(slot, x, y, String::new(), 0.0, 1.0, level);
            s.button.rect.width = if master { 310 } else { 150 };
            s.button.text = volume_label(name, level, lang);
            self.sliders.push(((*name).to_owned(), s));
            slot += if master { 2 } else { 1 };
        }
        // An unchanged layout keeps the grabbed slider; a resize builds fresh buttons, as
        // `initGui` does.
        if let Some((index, rect, position)) = drag {
            if let Some((_, s)) = self.sliders.get_mut(index) {
                if s.button.rect == rect {
                    s.set_position(position);
                    s.dragging = true;
                }
            }
        }
        self.done = Button::new(
            200,
            w / 2 - 100,
            h / 6 + 168,
            lang.tr("gui.done").to_owned(),
        );
    }

    /// Handle a click.
    pub fn mouse_clicked(
        &mut self,
        x: i32,
        y: i32,
        button: i32,
        settings: &mut GameSettings,
        lang: &Language,
    ) -> Action {
        if button != 0 {
            return Action::None;
        }
        // java: `GuiScreenOptionsSounds.actionPerformed` id 200 —
        // `mc.gameSettings.saveOptions(); mc.displayGuiScreen(this.field_146505_f)`. The save is
        // the caller's; what the screen reports is "leave".
        if self.done.mouse_pressed(x, y) {
            self.events.push(SoundOptionsEvent::SaveSettings);
            return Action::Back;
        }
        for (index, (name, s)) in self.sliders.iter_mut().enumerate() {
            // java: this slider overrides `playPressSound` with an empty body; its only click
            // is the one `mouseReleased` plays. `Slider::mouse_pressed` marks the widget hit the
            // driver turns into the default press sound, so test the hit without that marker.
            if s.button.enabled && s.button.visible && s.button.contains(x, y) {
                s.set_position(snap_level(s.position_for(x)));
                s.dragging = true;
                self.last_slider = Some(index);
                let level = s.position();
                settings.sound_levels.insert(name.clone(), level);
                s.button.text = volume_label(name, level, lang);
                self.events.push(SoundOptionsEvent::ApplyVolume {
                    category: name.clone(),
                    level,
                });
                self.events.push(SoundOptionsEvent::SaveSettings);
                return Action::None;
            }
        }
        Action::None
    }

    /// Continue a drag.
    pub fn mouse_dragged(&mut self, x: i32, settings: &mut GameSettings, lang: &Language) {
        for (name, s) in &mut self.sliders {
            if s.dragging {
                s.mouse_dragged(x);
                s.set_position(snap_level(s.position()));
                let level = s.position();
                settings.sound_levels.insert(name.clone(), level);
                s.button.text = volume_label(name, level, lang);
                self.events.push(SoundOptionsEvent::ApplyVolume {
                    category: name.clone(),
                    level,
                });
                self.events.push(SoundOptionsEvent::SaveSettings);
            }
        }
    }

    /// End any drag.
    pub fn mouse_released(&mut self) {
        for (_, s) in &mut self.sliders {
            if s.dragging {
                self.events.push(SoundOptionsEvent::PlayReleaseClick);
            }
            s.mouse_released();
        }
    }

    /// Fine adjustment (X1, item 33): Left/Right (Shift: x10) move
    /// the volume under the pointer, else the one pressed last, one percent, with the same
    /// side effects as a drag (live volume, then save). Returns whether a volume changed.
    pub fn slider_key(
        &mut self,
        key: i32,
        shift: bool,
        settings: &mut GameSettings,
        lang: &Language,
    ) -> bool {
        let Some(steps) = super::arrow_steps(key, shift) else {
            return false;
        };
        let hovered = self
            .sliders
            .iter()
            .position(|(_, s)| s.button.visible && s.button.is_mouse_over());
        let Some(index) = hovered.or(self.last_slider) else {
            return false;
        };
        let Some((name, s)) = self.sliders.get_mut(index) else {
            return false;
        };
        s.step_position(steps, VOLUME_STEPS);
        let level = s.position();
        settings.sound_levels.insert(name.clone(), level);
        s.button.text = volume_label(name, level, lang);
        self.events.push(SoundOptionsEvent::ApplyVolume {
            category: name.clone(),
            level,
        });
        self.events.push(SoundOptionsEvent::SaveSettings);
        self.last_slider = Some(index);
        true
    }

    /// Draw.
    pub fn draw(
        &mut self,
        d: &mut DrawList,
        font: &mut FontRenderer,
        ctx: &ScreenCtx,
        mouse: (i32, i32),
        lang: &Language,
    ) {
        draw_world_background(d, ctx.width, ctx.height, self.in_world);
        draw_centered_string(
            d,
            font,
            lang.tr("options.sounds.title"),
            ctx.width / 2,
            15,
            TITLE_COLOR,
        );
        for (_, s) in &mut self.sliders {
            s.draw(d, font, mouse.0, mouse.1);
        }
        self.done.draw(d, font, mouse.0, mouse.1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::widget::ScaledResolution;

    fn ctx() -> ScreenCtx {
        ScreenCtx::new(ScaledResolution::new(854, 480, 1, false), 0.0)
    }

    #[test]
    fn master_takes_two_slots_and_is_three_hundred_and_ten_wide() {
        let mut s = SoundOptionsScreen::new(false);
        let settings = GameSettings::default();
        s.init(&ctx(), &settings, &Language::new("en_US"));
        assert_eq!(s.sliders.len(), SOUND_CATEGORIES.len());
        assert_eq!(s.sliders[0].0, "master");
        assert_eq!(s.sliders[0].1.button.rect.width, 310);
        assert_eq!(s.sliders[0].1.button.rect.x, 854 / 2 - 155);
        assert_eq!(s.sliders[0].1.button.rect.y, 480 / 6 - 12);
        // The next category starts on row 1, column 0.
        assert_eq!(s.sliders[1].1.button.rect.x, 854 / 2 - 155);
        assert_eq!(s.sliders[1].1.button.rect.y, 480 / 6 - 12 + 24);
        assert_eq!(s.sliders[1].1.button.rect.width, 150);
        // And the one after that is column 1 of the same row.
        assert_eq!(s.sliders[2].1.button.rect.x, 854 / 2 - 155 + 160);
        assert_eq!(s.sliders[2].1.button.rect.y, s.sliders[1].1.button.rect.y);
    }

    #[test]
    fn zero_volume_reads_off_and_everything_else_truncates_to_a_percentage() {
        let mut l = Language::new("en_US");
        l.insert("options.off", "OFF");
        l.insert("soundCategory.music", "Music");
        assert_eq!(volume_label("music", 0.0, &l), "Music: OFF");
        assert_eq!(volume_label("music", 1.0, &l), "Music: 100%");
        // 0.335 * 100 = 33.5 → truncated to 33, not rounded to 34.
        assert_eq!(volume_label("music", 0.335, &l), "Music: 33%");
    }

    #[test]
    fn dragging_a_slider_writes_the_level_back_and_relabels_it() {
        let mut s = SoundOptionsScreen::new(false);
        let mut settings = GameSettings::default();
        let mut l = Language::new("en_US");
        l.insert("options.off", "OFF");
        s.init(&ctx(), &settings, &l);
        let rect = s.sliders[0].1.button.rect;
        assert_eq!(
            s.mouse_clicked(rect.x, rect.y + 5, 0, &mut settings, &l),
            Action::None
        );
        assert_eq!(settings.sound_level("master"), 0.0);
        assert!(s.sliders[0].1.button.text.ends_with("OFF"));
        assert_eq!(
            s.take_events(),
            vec![
                SoundOptionsEvent::ApplyVolume {
                    category: "master".to_owned(),
                    level: 0.0,
                },
                SoundOptionsEvent::SaveSettings,
            ]
        );
        s.mouse_dragged(rect.right(), &mut settings, &l);
        assert_eq!(settings.sound_level("master"), 1.0);
        assert!(s.sliders[0].1.button.text.ends_with("100%"));
        assert_eq!(
            s.take_events(),
            vec![
                SoundOptionsEvent::ApplyVolume {
                    category: "master".to_owned(),
                    level: 1.0,
                },
                SoundOptionsEvent::SaveSettings,
            ]
        );
        s.mouse_released();
        assert_eq!(s.take_events(), vec![SoundOptionsEvent::PlayReleaseClick]);
    }

    /// java: `mouseDragged` calls `setSoundLevel` and `saveOptions` on every rendered drag
    /// callback, even when the cursor has not moved since the previous frame.
    #[test]
    fn every_drag_callback_emits_live_apply_then_save_even_without_motion() {
        let mut s = SoundOptionsScreen::new(false);
        let mut settings = GameSettings::default();
        let l = Language::new("en_US");
        s.init(&ctx(), &settings, &l);
        let rect = s.sliders[0].1.button.rect;
        s.mouse_clicked(rect.x + 40, rect.y + 5, 0, &mut settings, &l);
        s.take_events();

        s.mouse_dragged(rect.x + 40, &mut settings, &l);
        let first = s.take_events();
        s.mouse_dragged(rect.x + 40, &mut settings, &l);
        let second = s.take_events();

        assert_eq!(first.len(), 2);
        assert_eq!(second, first);
        assert!(matches!(first[0], SoundOptionsEvent::ApplyVolume { .. }));
        assert_eq!(first[1], SoundOptionsEvent::SaveSettings);
    }

    #[test]
    fn release_click_is_only_emitted_for_a_grabbed_slider() {
        let mut s = SoundOptionsScreen::new(false);
        let mut settings = GameSettings::default();
        let l = Language::new("en_US");
        s.init(&ctx(), &settings, &l);

        s.mouse_released();
        assert!(s.take_events().is_empty());

        let rect = s.sliders[0].1.button.rect;
        assert_eq!(
            s.mouse_clicked(rect.x + 4, rect.y + 4, 1, &mut settings, &l),
            Action::None
        );
        s.mouse_released();
        assert!(s.take_events().is_empty());

        s.mouse_clicked(rect.x + 4, rect.y + 4, 0, &mut settings, &l);
        s.take_events();
        s.mouse_released();
        assert_eq!(s.take_events(), vec![SoundOptionsEvent::PlayReleaseClick]);
        s.mouse_released();
        assert!(s.take_events().is_empty());
    }

    /// java: id 200 is `saveOptions()` then `displayGuiScreen(parent)`.
    #[test]
    fn done_returns_to_the_parent_screen() {
        let mut s = SoundOptionsScreen::new(false);
        let mut settings = GameSettings::default();
        let l = Language::new("en_US");
        s.init(&ctx(), &settings, &l);
        let (x, y) = (854 / 2, 480 / 6 + 168 + 1);
        assert_eq!(s.mouse_clicked(x, y, 0, &mut settings, &l), Action::Back);
        assert_eq!(s.take_events(), vec![SoundOptionsEvent::SaveSettings]);
    }

    /// `Client::build_ui` calls `init` before every frame. The drag latch set by the press must
    /// survive that, as `field_146155_p` survives frames in vanilla, or a volume can only be
    /// clicked to a spot and never dragged. A real resize still rebuilds the buttons and drops
    /// the drag, as `initGui` does.
    #[test]
    fn a_volume_drag_survives_the_per_frame_relayout() {
        let mut s = SoundOptionsScreen::new(false);
        let mut settings = GameSettings::default();
        let l = Language::new("en_US");
        s.init(&ctx(), &settings, &l);
        let rect = s.sliders[0].1.button.rect;
        let x_at = |fraction: f32| rect.x + 4 + ((rect.width - 8) as f32 * fraction) as i32;

        s.mouse_clicked(x_at(0.25), rect.y + 5, 0, &mut settings, &l);
        for fraction in [0.5, 0.75] {
            // One frame: the host relays the screen out, then feeds the held button's drag.
            s.init(&ctx(), &settings, &l);
            s.mouse_dragged(x_at(fraction), &mut settings, &l);
            let level = settings.sound_level("master");
            assert!((level - fraction).abs() < 0.01, "{fraction}: {level}");
        }
        s.init(&ctx(), &settings, &l);
        assert!(s.sliders[0].1.dragging);
        s.mouse_released();
        s.init(&ctx(), &settings, &l);
        s.mouse_dragged(x_at(0.1), &mut settings, &l);
        assert!(
            (settings.sound_level("master") - 0.75).abs() < 0.01,
            "released"
        );

        s.mouse_clicked(x_at(0.25), rect.y + 5, 0, &mut settings, &l);
        s.init(
            &ScreenCtx::new(ScaledResolution::new(640, 480, 1, false), 0.0),
            &settings,
            &l,
        );
        assert!(
            s.sliders.iter().all(|(_, slider)| !slider.dragging),
            "a resize builds fresh buttons"
        );
    }

    /// java: this `GuiButton` overrides `playPressSound` with an empty body, so a press must not
    /// mark the widget hit that the host turns into the default `gui.button.press`; the one
    /// click is `PlayReleaseClick`. The Done button is an ordinary `GuiButton` and still does.
    #[test]
    fn a_volume_press_is_silent_and_only_the_release_clicks() {
        let mut s = SoundOptionsScreen::new(false);
        let mut settings = GameSettings::default();
        let l = Language::new("en_US");
        s.init(&ctx(), &settings, &l);
        let rect = s.sliders[0].1.button.rect;

        let (action, hit) = crate::widget::capture_button_press(|| {
            s.mouse_clicked(rect.x + 40, rect.y + 5, 0, &mut settings, &l)
        });
        assert_eq!(action, Action::None);
        assert!(
            !hit,
            "a volume press must not request the default press sound"
        );
        assert!(s.sliders[0].1.dragging, "the press still grabs the slider");
        s.take_events();
        s.mouse_released();
        assert_eq!(s.take_events(), vec![SoundOptionsEvent::PlayReleaseClick]);

        let (action, hit) = crate::widget::capture_button_press(|| {
            s.mouse_clicked(854 / 2, 480 / 6 + 168 + 1, 0, &mut settings, &l)
        });
        assert_eq!(action, Action::Back);
        assert!(hit, "control: Done is an ordinary sounding button");
    }

    /// A volume snaps to the whole percent its label
    /// shows, every percent 0..=100 has one value, and Left/Right step the last-pressed volume
    /// one percent with a drag's side effects.
    #[test]
    fn volumes_snap_to_whole_percents_and_the_arrows_step_one_percent() {
        let mut l = Language::new("en_US");
        l.insert("options.off", "OFF");
        l.insert("soundCategory.master", "Master Volume");
        for k in 0..=100 {
            let level = snap_level(k as f32 / 100.0 + 0.004);
            let label = volume_label("master", level, &l);
            let want = if k == 0 {
                "Master Volume: OFF".to_owned()
            } else {
                format!("Master Volume: {k}%")
            };
            assert_eq!(label, want, "{k}");
        }
        let mut s = SoundOptionsScreen::new(false);
        let mut settings = GameSettings::default();
        s.init(&ctx(), &settings, &l);
        assert!(
            !s.slider_key(crate::widget::keys::RIGHT, false, &mut settings, &l),
            "no slider pressed or hovered yet"
        );
        let rect = s.sliders[0].1.button.rect;
        s.mouse_clicked(rect.x, rect.y + 5, 0, &mut settings, &l);
        s.mouse_released();
        s.take_events();
        for k in 1..=100 {
            assert!(s.slider_key(crate::widget::keys::RIGHT, false, &mut settings, &l));
            let want = format!("Master Volume: {k}%");
            assert_eq!(
                volume_label("master", settings.sound_level("master"), &l),
                want
            );
        }
        assert_eq!(
            s.take_events()[..2],
            [
                SoundOptionsEvent::ApplyVolume {
                    category: "master".to_owned(),
                    level: 0.01,
                },
                SoundOptionsEvent::SaveSettings,
            ]
        );
        s.slider_key(crate::widget::keys::LEFT, true, &mut settings, &l);
        assert_eq!(settings.sound_level("master"), 0.9);
    }
}
