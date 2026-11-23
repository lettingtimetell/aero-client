//! The demo-world help screen.
//!
//! java: `client/gui/GuiScreenDemo.java`

use mc_assets::Language;
use mc_render::input::{keys, KeyBindings};

use crate::font::FontRenderer;
use crate::screen_stack::{Screen, ScreenCtx};
use crate::screens::common::draw_world_background;
use crate::screens::Action;
use crate::widget::draw::{DrawList, UiTexture};
use crate::widget::Button;

/// The URL opened by vanilla's Buy button.
pub const DEMO_STORE_URL: &str = "http://www.minecraft.net/store?source=demo";
const BACKGROUND: &str = "textures/gui/demo_background.png";
const BACKGROUND_WIDTH: i32 = 248;
const BACKGROUND_HEIGHT: i32 = 166;
const TITLE_COLOR: i32 = 2_039_583;
const BODY_COLOR: i32 = 5_197_647;

mod ids {
    pub const BUY: i32 = 1;
    pub const LATER: i32 = 2;
}

/// java: `GuiScreenDemo`.
#[derive(Debug, Default)]
pub struct DemoScreen {
    buttons: Vec<Button>,
}

impl DemoScreen {
    /// Construct an unlaid-out demo help screen.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Lay out the two 114x20 buttons. java: `initGui`, including `i = -16`.
    pub fn init(&mut self, ctx: &ScreenCtx, lang: &Language) {
        let y = ctx.height / 2 + 46;
        self.buttons = vec![
            Button::sized(
                ids::BUY,
                ctx.width / 2 - 116,
                y,
                114,
                20,
                lang.tr("demo.help.buy"),
            ),
            Button::sized(
                ids::LATER,
                ctx.width / 2 + 2,
                y,
                114,
                20,
                lang.tr("demo.help.later"),
            ),
        ];
    }

    /// Translate one of reason 5's chat hints with the live key bindings.
    #[must_use]
    pub fn chat_help(value: f32, bindings: &KeyBindings, lang: &Language) -> Option<String> {
        match value {
            101.0 => {
                let names = binding_names(
                    bindings,
                    &[keys::FORWARD, keys::LEFT, keys::BACK, keys::RIGHT],
                );
                Some(lang.format(
                    "demo.help.movement",
                    &[&names[0], &names[1], &names[2], &names[3]],
                ))
            }
            102.0 => {
                let name = binding_name(bindings, keys::JUMP);
                Some(lang.format("demo.help.jump", &[&name]))
            }
            103.0 => {
                let name = binding_name(bindings, keys::INVENTORY);
                Some(lang.format("demo.help.inventory", &[&name]))
            }
            _ => None,
        }
    }

    /// Handle one mouse press against the Buy and Later buttons.
    pub fn mouse_clicked(&mut self, x: i32, y: i32, button: i32) -> Action {
        if button != 0 {
            return Action::None;
        }
        for candidate in &mut self.buttons {
            if candidate.mouse_pressed(x, y) {
                return match candidate.id {
                    ids::BUY => {
                        // Java disables the button before attempting Desktop.browse.
                        candidate.enabled = false;
                        Action::OpenUrl(DEMO_STORE_URL.to_owned())
                    }
                    ids::LATER => Action::CloseAll,
                    _ => Action::None,
                };
            }
        }
        Action::None
    }

    /// Draw the live-world backdrop, demo panel, rebound help text, and buttons.
    pub fn draw(
        &mut self,
        d: &mut DrawList,
        font: &mut FontRenderer,
        ctx: &ScreenCtx,
        mouse: (i32, i32),
        bindings: &KeyBindings,
        lang: &Language,
    ) {
        draw_world_background(d, ctx.width, ctx.height, true);
        let left = (ctx.width - BACKGROUND_WIDTH) / 2;
        let top = (ctx.height - BACKGROUND_HEIGHT) / 2;
        d.blit(
            &UiTexture::gui(BACKGROUND),
            left,
            top,
            0,
            0,
            BACKGROUND_WIDTH,
            BACKGROUND_HEIGHT,
        );

        let x = left + 10;
        let title_y = top + 8;
        font.draw_string(
            d,
            lang.tr("demo.help.title"),
            x as f32,
            title_y as f32,
            TITLE_COLOR,
            false,
        );
        let body_y = title_y + 12;
        let movement = binding_names(
            bindings,
            &[keys::FORWARD, keys::LEFT, keys::BACK, keys::RIGHT],
        );
        let movement = lang.format(
            "demo.help.movementShort",
            &[&movement[0], &movement[1], &movement[2], &movement[3]],
        );
        let jump = binding_name(bindings, keys::JUMP);
        let inventory = binding_name(bindings, keys::INVENTORY);
        let lines = [
            (movement, body_y),
            (lang.tr("demo.help.movementMouse").to_owned(), body_y + 12),
            (lang.format("demo.help.jump", &[&jump]), body_y + 24),
            (
                lang.format("demo.help.inventory", &[&inventory]),
                body_y + 36,
            ),
        ];
        for (text, y) in lines {
            font.draw_string(d, &text, x as f32, y as f32, BODY_COLOR, false);
        }
        font.draw_split_string(
            d,
            lang.tr("demo.help.fullWrapped"),
            x,
            body_y + 68,
            218,
            TITLE_COLOR,
        );
        for button in &mut self.buttons {
            button.draw(d, font, mouse.0, mouse.1);
        }
    }
}

fn binding_name(bindings: &KeyBindings, description: &str) -> String {
    bindings.get(description).map_or_else(
        || "NONE".to_owned(),
        |binding| binding.binding().display_name(),
    )
}

fn binding_names<const N: usize>(bindings: &KeyBindings, descriptions: &[&str; N]) -> [String; N] {
    descriptions.map(|description| binding_name(bindings, description))
}

impl Screen for DemoScreen {
    fn name(&self) -> &'static str {
        "demo"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::widget::ScaledResolution;
    use mc_render::input::Input;
    use winit::keyboard::KeyCode;

    fn ctx() -> ScreenCtx {
        ScreenCtx::new(ScaledResolution::new(854, 480, 1, false), 0.0)
    }

    fn language() -> Language {
        let mut lang = Language::new("en_US");
        lang.merge_lang_str(
            "demo.help.buy=Buy Now!\n\
             demo.help.later=Purchase Later\n\
             demo.help.movement=Move with %s, %s, %s, %s\n\
             demo.help.jump=Jump with %s\n\
             demo.help.inventory=Inventory with %s\n",
        );
        lang
    }

    fn screen() -> DemoScreen {
        let mut screen = DemoScreen::new();
        screen.init(&ctx(), &language());
        screen
    }

    #[test]
    fn buttons_match_vanilla_and_produce_terminal_actions() {
        let mut screen = screen();
        let buy = screen
            .buttons
            .iter()
            .find(|button| button.id == ids::BUY)
            .unwrap();
        let later = screen
            .buttons
            .iter()
            .find(|button| button.id == ids::LATER)
            .unwrap();
        assert_eq!(
            (buy.rect.x, buy.rect.y, buy.rect.width, buy.rect.height),
            (311, 286, 114, 20)
        );
        assert_eq!(
            (
                later.rect.x,
                later.rect.y,
                later.rect.width,
                later.rect.height
            ),
            (429, 286, 114, 20)
        );

        assert_eq!(
            screen.mouse_clicked(312, 287, 0),
            Action::OpenUrl(DEMO_STORE_URL.to_owned())
        );
        assert_eq!(screen.mouse_clicked(312, 287, 0), Action::None);
        assert_eq!(screen.mouse_clicked(430, 287, 0), Action::CloseAll);
    }

    #[test]
    fn chat_hints_use_live_rebound_keys_and_other_values_are_inert() {
        let lang = language();
        let mut bindings = KeyBindings::vanilla();
        bindings.set_binding(keys::FORWARD, Input::Key(KeyCode::ArrowUp));
        bindings.set_binding(keys::LEFT, Input::Key(KeyCode::ArrowLeft));
        bindings.set_binding(keys::BACK, Input::Key(KeyCode::ArrowDown));
        bindings.set_binding(keys::RIGHT, Input::Key(KeyCode::ArrowRight));
        bindings.set_binding(keys::JUMP, Input::Key(KeyCode::KeyJ));
        bindings.set_binding(keys::INVENTORY, Input::Key(KeyCode::KeyI));

        assert_eq!(
            DemoScreen::chat_help(101.0, &bindings, &lang).as_deref(),
            Some("Move with UP, LEFT, DOWN, RIGHT")
        );
        assert_eq!(
            DemoScreen::chat_help(102.0, &bindings, &lang).as_deref(),
            Some("Jump with J")
        );
        assert_eq!(
            DemoScreen::chat_help(103.0, &bindings, &lang).as_deref(),
            Some("Inventory with I")
        );
        assert!(DemoScreen::chat_help(104.0, &bindings, &lang).is_none());
    }
}
