//! Independent container and bottom-HUD sizing. Input and rendering share one resolution.
use std::cell::Cell;

use mc_mods::ModSettings;
use mc_ui::widget::ScaledResolution;

use super::super::{Client, Ui};

/// Cached choice indices; a settings-generation check keeps catalog scans out of each draw.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct GuiScaleSnapshot {
    seen: u64,
    enabled: bool,
    inventory: usize,
    hotbar: usize,
}

impl GuiScaleSnapshot {
    fn read(cache: &Cell<Self>, settings: &ModSettings) -> Self {
        let mut value = cache.get();
        if value.seen != settings.generation() {
            value = Self {
                seen: settings.generation(),
                enabled: settings.enabled("independent_gui"),
                inventory: settings.choice("independent_gui", "inventory_scale"),
                hotbar: settings.choice("independent_gui", "hotbar_scale"),
            };
            cache.set(value);
        }
        value
    }

    fn scale(self, choice: usize, global: i32) -> i32 {
        if !self.enabled || choice == 0 {
            global
        } else {
            // Auto=0, Small=1, Normal=2, Large=3 in vanilla's scale constructor.
            choice.saturating_sub(1).min(3) as i32
        }
    }
}

impl Client {
    /// Layout and pointer space for the currently open screen. The global HUD stays separate.
    pub(crate) fn screen_resolution(&self, width: u32, height: u32) -> ScaledResolution {
        let snapshot = GuiScaleSnapshot::read(&self.gui_scaling, &self.mods.settings);
        let scale = if matches!(self.ui, Ui::Container(_) | Ui::Creative(_)) {
            snapshot.scale(snapshot.inventory, self.settings.gui_scale)
        } else {
            self.settings.gui_scale
        };
        ScaledResolution::new(
            width,
            height,
            scale,
            self.assets.language.is_unicode() || self.settings.force_unicode_font,
        )
    }

    /// Resolution for the hotbar and its attached stats, captions and spectator bar.
    pub(crate) fn hotbar_resolution(&self, width: u32, height: u32) -> ScaledResolution {
        let snapshot = GuiScaleSnapshot::read(&self.gui_scaling, &self.mods.settings);
        ScaledResolution::new(
            width,
            height,
            snapshot.scale(snapshot.hotbar, self.settings.gui_scale),
            self.assets.language.is_unicode() || self.settings.force_unicode_font,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mc_mods::OptionValue;

    #[test]
    fn independent_choices_follow_defaults_toggle_and_settings_changes() {
        let mut settings = ModSettings::new();
        let cache = Cell::default();
        settings.set_value("independent_gui", "inventory_scale", OptionValue::Choice(1));
        settings.set_value("independent_gui", "hotbar_scale", OptionValue::Choice(2));
        let off = GuiScaleSnapshot::read(&cache, &settings);
        assert_eq!(off.scale(off.inventory, 3), 3);
        assert_eq!(off.scale(off.hotbar, 3), 3);
        settings.set_enabled("independent_gui", true);
        let on = GuiScaleSnapshot::read(&cache, &settings);
        assert_eq!(on.scale(on.inventory, 3), 0);
        assert_eq!(on.scale(on.hotbar, 3), 1);
        assert_eq!(on.scale(0, 2), 2);
        settings.set_enabled("independent_gui", false);
        let disabled = GuiScaleSnapshot::read(&cache, &settings);
        assert_eq!(disabled.scale(disabled.hotbar, 2), 2);
    }
    #[test]
    fn independent_container_input_click_drag_and_resize_use_the_draw_resolution() {
        use crate::app::config::{ClientConfig, LoginMode};
        use mc_protocol::item::ItemStack;
        use mc_ui::screen_stack::{Screen, ScreenCtx};
        use mc_ui::screens::container::ContainerScreen;
        use winit::event::MouseButton;

        let dir = tempfile::tempdir().unwrap();
        let config = ClientConfig {
            game_dir: dir.path().to_path_buf(),
            login: LoginMode::Offline("ScaleTest".into()),
            ..ClientConfig::default()
        }
        .with_discovered_assets();
        if !config.version_jar.as_ref().is_some_and(|p| p.is_file()) {
            eprintln!("skip: no vanilla jar in this checkout");
            return;
        }
        let mut client = Client::new(config, &mut |_, _| {}).unwrap();
        client.settings.gui_scale = 3;
        client.mods.settings.set_enabled("independent_gui", true);
        client
            .mods
            .settings
            .set_choice("independent_gui", "inventory_scale", 1);
        client
            .mods
            .settings
            .set_choice("independent_gui", "hotbar_scale", 2);
        let mut screen = ContainerScreen::chest(1, 3, "Scale test");
        screen.container.set_slot_contents(
            0,
            Some(ItemStack {
                item_id: 1,
                count: 8,
                damage: 0,
                nbt: None,
            }),
        );
        client.ui = Ui::Container(Box::new(screen));
        let (width, height) = (1921, 1081);
        let resolution = client.screen_resolution(width, height);
        assert_eq!(resolution.scale_factor(), 4);
        assert_eq!(client.resolution(width, height).scale_factor(), 3);
        assert_eq!(client.hotbar_resolution(width, height).scale_factor(), 1);
        let ctx = ScreenCtx::new(resolution, 0.0);
        let mut layout = ContainerScreen::chest(1, 3, "Scale test");
        layout.init(&ctx);
        let physical_slot = |id: usize| {
            let slot = layout.container.slots()[id];
            let gx = layout.gui_left() + slot.x + 8;
            let gy = layout.gui_top() + slot.y + 8;
            (
                (gx as f64 + 0.5) * width as f64 / ctx.width as f64,
                (gy as f64 + 0.5) * height as f64 / ctx.height as f64,
            )
        };
        let mut input = mc_render::InputState::new();
        // A closing mod menu can still be alive after a container has opened underneath it.
        client.mods.menu_live = true;
        let (x, y) = physical_slot(0);
        input.on_cursor_moved(x, y);
        input.on_mouse_button(MouseButton::Left, true);
        client.handle_screen_input(&mut input, width, height);
        let global = client.resolution(width, height);
        assert_eq!(
            client.mods.menu_gui,
            (
                (global.scaled_width() as f32, global.scaled_height() as f32),
                global.scale_factor() as f32,
            ),
            "closing mod-menu geometry remains in global HUD space"
        );
        let Ui::Container(screen) = &client.ui else {
            panic!("container closed");
        };
        assert_eq!(
            screen.container.cursor().map(|s| s.count),
            Some(8),
            "first click before any draw hits the visible slot"
        );
        assert_eq!(
            screen.slot_at_position(client.cursor.0, client.cursor.1),
            Some(0)
        );
        input.on_mouse_button(MouseButton::Left, false);
        client.handle_screen_input(&mut input, width, height);
        let (x, y) = physical_slot(1);
        input.on_cursor_moved(x, y);
        input.on_mouse_button(MouseButton::Left, true);
        client.handle_screen_input(&mut input, width, height);
        // java: the press only arms the drag. `GuiScreen.handleMouseInput` reaches
        // `mouseClickMove` from LWJGL move events alone (`:631-635`), so slot 1 joins the drag
        // only through movement over it after the press, as a real drag's first pixels are. One
        // physical pixel stays inside slot 1 at this scale.
        input.on_cursor_moved(x + 1.0, y);
        client.handle_screen_input(&mut input, width, height);
        let (x, y) = physical_slot(2);
        input.on_cursor_moved(x, y);
        client.handle_screen_input(&mut input, width, height);
        input.on_mouse_button(MouseButton::Left, false);
        client.handle_screen_input(&mut input, width, height);
        let Ui::Container(screen) = &client.ui else {
            panic!("container closed");
        };
        assert_eq!(screen.container.slot_stack(1).map(|s| s.count), Some(4));
        assert_eq!(screen.container.slot_stack(2).map(|s| s.count), Some(4));
        assert!(screen.container.cursor().is_none());

        // A still click stays one ordinary click at the independent scale: pick slot 2 up, then
        // a still shift-click with that stack on empty slot 0 is `mouseReleased`'s mode 1, which
        // transfers nothing (a drag would have placed the stack there).
        let (x, y) = physical_slot(2);
        input.on_cursor_moved(x, y);
        client.handle_screen_input(&mut input, width, height);
        input.on_mouse_button(MouseButton::Left, true);
        client.handle_screen_input(&mut input, width, height);
        input.on_mouse_button(MouseButton::Left, false);
        client.handle_screen_input(&mut input, width, height);
        let (x, y) = physical_slot(0);
        input.on_cursor_moved(x, y);
        client.handle_screen_input(&mut input, width, height);
        client
            .keyboard
            .push(winit::keyboard::KeyCode::ShiftLeft, true, false, None);
        input.on_mouse_button(MouseButton::Left, true);
        client.handle_screen_input(&mut input, width, height);
        client.handle_screen_input(&mut input, width, height);
        input.on_mouse_button(MouseButton::Left, false);
        client.handle_screen_input(&mut input, width, height);
        client
            .keyboard
            .push(winit::keyboard::KeyCode::ShiftLeft, false, false, None);
        let Ui::Container(screen) = &client.ui else {
            panic!("container closed");
        };
        assert_eq!(screen.container.slot_stack(0), None);
        assert_eq!(screen.container.cursor().map(|s| s.count), Some(4));

        client.handle_screen_input(&mut input, 800, 600);
        let resized = client.screen_resolution(800, 600);
        assert_eq!(resized.scale_factor(), 2);
        let Ui::Container(screen) = &client.ui else {
            panic!("container closed");
        };
        assert_eq!(screen.gui_left(), (resized.scaled_width() - 176) / 2);
        client.ui = Ui::InGame;
        assert_eq!(
            client.screen_resolution(width, height),
            client.resolution(width, height),
            "closing a container restores the normal screen space"
        );
        assert_eq!(client.hotbar_resolution(width, height).scale_factor(), 1);
        client.mods.settings.set_enabled("independent_gui", false);
        assert_eq!(
            client.screen_resolution(width, height),
            client.resolution(width, height)
        );
        assert_eq!(
            client.hotbar_resolution(width, height),
            client.resolution(width, height)
        );
    }
}
