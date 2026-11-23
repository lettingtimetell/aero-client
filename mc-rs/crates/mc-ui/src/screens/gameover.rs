//! The death screen.
//!
//! java: `client/gui/GuiGameOver.java`
//!
//! # Why this file exists at all
//!
//! It was simply missing. S16's task list names "Pause menu, disconnect screen, connecting/login
//! progress" and `GuiGameOver` is in none of those, so a player who died on a real server got no
//! screen, no Respawn button, and — because 1.8.9's server holds a dead client in place until it
//! receives `C16PacketClientStatus(PERFORM_RESPAWN)` — no way back into the world at all. Found
//! by a human play-testing against a live 1.8 server, not by a test.
//!
//! # The two vanilla quirks that matter
//!
//! * **The buttons are dead for the first second.** `initGui` disables *every* button it just
//!   added, and `updateScreen` re-enables them when `enableButtonsTimer` reaches exactly `20`.
//!   Vanilla does this so the click that killed you cannot also dismiss the screen. A port that
//!   enables them immediately is not merely different — it makes the screen useless at exactly
//!   the moment it appears, which is what the timer exists to prevent.
//! * **The screen appears via `displayGuiScreen(null)`.** `Minecraft.runTick:1773` asks for *no*
//!   screen when the player's health hits zero, and `displayGuiScreen` substitutes a
//!   `GuiGameOver` because `thePlayer.getHealth() <= 0.0F`. That indirection is reproduced in
//!   `mc-client`'s driver rather than here, but it is why "close every screen" is what raises
//!   this one.
//!
//! Hardcore mode replaces Respawn with a single "Delete world" / "Leave server" button. The multiplayer-only rule makes this client multiplayer-only, so the hardcore branch is the `deathScreen.leaveServer`
//! one; it is reachable through [`GameOverScreen::new`]'s `hardcore` flag, which S14's
//! `JoinGame.hardcore_mode` supplies.

use mc_assets::Language;
use mc_math::JavaRandom;

use crate::font::FontRenderer;
use crate::screen_stack::{Screen, ScreenAction, ScreenCtx};
use crate::screens::common::TITLE_COLOR;
use crate::screens::yes_no::YesNoScreen;
use crate::screens::Action;
use crate::widget::draw::{DrawList, UiTexture, OPTIONS_BACKGROUND};
use crate::widget::keys::Modifiers;
use crate::widget::transform::Transform;
use crate::widget::{draw_centered_string, Button};

/// java: `drawGradientRect(0, 0, width, height, 1615855616, -1602211792)` — the top colour.
pub const DEATH_GRADIENT_TOP: u32 = 0x6050_0000;
/// The bottom colour of the same gradient. `-1602211792` as an unsigned ARGB word.
pub const DEATH_GRADIENT_BOTTOM: u32 = 0xA080_3030;

/// How many ticks the buttons stay disabled. java: `enableButtonsTimer == 20`
pub const ENABLE_BUTTONS_TICK: i32 = 20;

/// The width passed to `FontRenderer.listFormattedStringToWidth` by `GuiWinGame`.
/// java: `GuiWinGame.initGui`: `int i = 274`.
pub const WIN_GAME_TEXT_WIDTH: i32 = 274;
/// The default end-poem scroll speed. java: `field_146578_s = 0.5F`.
pub const WIN_GAME_SCROLL_SPEED: f32 = 0.5;
/// The number of blank lines inserted between the end poem and the credits.
/// java: `GuiWinGame.initGui`: `for (int k = 0; k < 8; ++k)`.
pub const WIN_GAME_SEPARATOR_LINES: usize = 8;

const WIN_GAME_LOGO_TEXTURE: &str = "textures/gui/title/minecraft.png";
const WIN_GAME_VIGNETTE_TEXTURE: &str = "textures/misc/vignette.png";
const WIN_GAME_OBFUSCATION_MARKER: &str = "§f§k§a§b§3";

/// The scrolling end poem and credits. java: `GuiWinGame`.
///
/// The resource files are parsed by the client because `FontRenderer` owns the exact
/// 1.8.9 wrapping routine. Keeping the resulting lines here makes the screen independent of
/// `ResourceManager` and straightforward to exercise without a GPU or a live connection.
#[derive(Debug, Clone)]
pub struct WinGameScreen {
    lines: Vec<String>,
    /// java: `field_146581_h`.
    scroll_ticks: i32,
    /// java: `field_146579_r` (`field_146582_i.size() * 12`).
    content_height: i32,
    width: i32,
    height: i32,
    /// Once the terminal action has been requested, no later tick may send another respawn.
    sent_respawn: bool,
}

impl WinGameScreen {
    /// Build a screen from already-loaded `texts/end.txt` and `texts/credits.txt` contents.
    ///
    /// `GuiWinGame` uses the session username in both files, expands tabs in the credits, and
    /// replaces the special obfuscated marker in the poem with a deterministic random run of
    /// `X` characters. The random seed is the literal one from vanilla's `initGui`.
    #[must_use]
    pub fn from_texts(
        end_text: &str,
        credits_text: &str,
        username: &str,
        font: &FontRenderer,
    ) -> Self {
        let mut lines = Vec::new();
        let mut random = JavaRandom::new(8_124_371);

        for line in end_text.lines() {
            let line = expand_end_line(line, username, &mut random);
            lines.extend(font.wrap_to_lines(&line, WIN_GAME_TEXT_WIDTH));
            // java: each source line gets one explicit separator, including an empty source line.
            lines.push(String::new());
        }
        lines.extend(std::iter::repeat_n(String::new(), WIN_GAME_SEPARATOR_LINES));
        for line in credits_text.lines() {
            let line = line.replace("PLAYERNAME", username).replace('\t', "    ");
            lines.extend(font.wrap_to_lines(&line, WIN_GAME_TEXT_WIDTH));
            lines.push(String::new());
        }

        Self::new(lines)
    }

    /// Build a screen from pre-expanded display lines. This is useful for tests and for callers
    /// that already own the text parsing boundary.
    #[must_use]
    pub fn new(lines: Vec<String>) -> Self {
        Self {
            content_height: i32::try_from(lines.len())
                .unwrap_or(i32::MAX / 12)
                .saturating_mul(12),
            lines,
            scroll_ticks: 0,
            width: 0,
            height: 0,
            sent_respawn: false,
        }
    }

    /// The final formatted lines, including blank separators.
    #[must_use]
    pub fn lines(&self) -> &[String] {
        &self.lines
    }

    /// Number of update ticks elapsed since the screen opened.
    #[must_use]
    pub const fn scroll_ticks(&self) -> i32 {
        self.scroll_ticks
    }

    /// The vanilla content height used by the terminal scroll calculation.
    #[must_use]
    pub const fn content_height(&self) -> i32 {
        self.content_height
    }

    /// Whether the screen has already requested the terminal respawn action.
    #[must_use]
    pub const fn sent_respawn(&self) -> bool {
        self.sent_respawn
    }

    /// Resize without resetting the poem position. java: `GuiScreen.setWorldAndResolution`
    /// calls `initGui`, but `GuiWinGame.initGui` only parses text when its list is null.
    pub fn resize(&mut self, width: i32, height: i32) {
        self.width = width;
        self.height = height;
    }

    /// Advance one client tick, returning whether `C16(PERFORM_RESPAWN)` should be sent now.
    /// java: `GuiWinGame.updateScreen`.
    pub fn tick(&mut self) -> bool {
        if self.sent_respawn || self.height <= 0 {
            return false;
        }
        self.scroll_ticks = self.scroll_ticks.saturating_add(1);
        let end_tick = self.end_tick(self.height);
        if (self.scroll_ticks as f32) > end_tick {
            self.sent_respawn = true;
            true
        } else {
            false
        }
    }

    fn end_tick(&self, height: i32) -> f32 {
        let distance = self
            .content_height
            .saturating_add(height.saturating_mul(2))
            .saturating_add(24);
        distance as f32 / WIN_GAME_SCROLL_SPEED
    }

    /// Draw the dirt scroll, logo, text and vignette used by `GuiWinGame.drawScreen`.
    pub fn draw(
        &mut self,
        d: &mut DrawList,
        font: &mut FontRenderer,
        ctx: &ScreenCtx,
        partial_ticks: f32,
    ) {
        let width = ctx.width;
        let height = ctx.height;
        self.width = width;
        self.height = height;

        let position = self.scroll_ticks as f32 + partial_ticks;
        let background_top = -position * 0.5 * WIN_GAME_SCROLL_SPEED;
        let background_bottom = height as f32 + background_top;
        let end_tick = self.end_tick(height);
        let mut shade = position * 0.02;
        let terminal_shade = (end_tick - 20.0 - position) * 0.005;
        if terminal_shade < shade {
            shade = terminal_shade;
        }
        if shade > 1.0 {
            shade = 1.0;
        }
        shade *= shade;
        shade = shade * 96.0 / 255.0;

        // java: `drawWinGameScreen` binds options_background and lets the sampler repeat.
        let background = UiTexture::gui(OPTIONS_BACKGROUND);
        d.quad(
            &background,
            [
                [0.0, 0.0],
                [0.0, height as f32],
                [width as f32, 0.0],
                [width as f32, height as f32],
            ],
            [
                [0.0, background_bottom * 0.015625],
                [0.0, background_top * 0.015625],
                [width as f32 * 0.015625, background_bottom * 0.015625],
                [width as f32 * 0.015625, background_top * 0.015625],
            ],
            [[shade, shade, shade, 1.0]; 4],
        );

        let translation_y = -position * WIN_GAME_SCROLL_SPEED;
        let logo_x = width / 2 - WIN_GAME_TEXT_WIDTH / 2;
        let logo_y = height + 50;
        let text_x = logo_x;
        let mut text_y = height + 250;
        let last_line = self.lines.len().saturating_sub(1);
        let mut final_line_shift = 0.0;

        // The logo and every line share the same GL translation. The final-line correction is
        // applied to the coordinates below, preserving Java's one-off late-scroll adjustment.
        d.with_transform(Transform::translate(0.0, translation_y), |d| {
            d.blit(
                &UiTexture::gui(WIN_GAME_LOGO_TEXTURE),
                logo_x,
                logo_y,
                0,
                0,
                155,
                44,
            );
            d.blit(
                &UiTexture::gui(WIN_GAME_LOGO_TEXTURE),
                logo_x + 155,
                logo_y,
                0,
                45,
                155,
                44,
            );

            for (index, line) in self.lines.iter().enumerate() {
                if index == last_line {
                    let final_y = text_y as f32 + translation_y - (height / 2 - 6) as f32;
                    if final_y < 0.0 {
                        final_line_shift = -final_y;
                    }
                }
                let y = text_y as f32 + final_line_shift;
                if y + 20.0 > 0.0 && y < height as f32 {
                    if let Some(centered) = line.strip_prefix("[C]") {
                        let x = text_x + (WIN_GAME_TEXT_WIDTH - font.string_width(centered)) / 2;
                        font.draw_string_with_shadow(d, centered, x as f32, y, TITLE_COLOR);
                    } else {
                        font.draw_string_with_shadow(d, line, text_x as f32, y, TITLE_COLOR);
                    }
                }
                text_y += 12;
            }
        });

        // java: the vignette is a full-screen post-texture pass. The ordinary alpha path is the
        // closest portable equivalent in the UI renderer; the texture itself carries the edge
        // alpha and remains a separate batch after the text.
        d.quad(
            &UiTexture::gui(WIN_GAME_VIGNETTE_TEXTURE),
            [
                [0.0, 0.0],
                [0.0, height as f32],
                [width as f32, 0.0],
                [width as f32, height as f32],
            ],
            [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
            [[1.0, 1.0, 1.0, 1.0]; 4],
        );
    }
}

fn expand_end_line(line: &str, username: &str, random: &mut JavaRandom) -> String {
    let mut expanded = line.replace("PLAYERNAME", username);
    while let Some(marker_start) = expanded.find(WIN_GAME_OBFUSCATION_MARKER) {
        let marker_end = marker_start + WIN_GAME_OBFUSCATION_MARKER.len();
        let run_length = usize::try_from(random.next_i32_bound(4) + 3).unwrap_or(3);
        let replacement = format!(
            "§f§k{}",
            "XXXXXXXX".chars().take(run_length).collect::<String>()
        );
        expanded.replace_range(marker_start..marker_end, &replacement);
    }
    expanded
}

/// Button ids, matching vanilla's.
mod ids {
    /// `deathScreen.respawn`.
    pub const RESPAWN: i32 = 0;
    /// `deathScreen.titleScreen`, or `deathScreen.leaveServer` in hardcore.
    pub const TITLE: i32 = 1;
}

/// The death screen. java: `GuiGameOver`
#[derive(Debug, Default)]
pub struct GameOverScreen {
    buttons: Vec<Button>,
    /// java: `enableButtonsTimer`
    timer: i32,
    /// Whether the world is hardcore, which removes the Respawn button entirely.
    hardcore: bool,
    /// `EntityPlayer.getScore()`, shown on the `deathScreen.score` line.
    score: i32,
    laid_out: bool,
    /// The non-hardcore title button's reusable vanilla `GuiYesNo` child.
    quit_confirmation: Option<YesNoScreen>,
    confirmation_message: String,
    confirmation_title: String,
    confirmation_respawn: String,
    width: i32,
    height: i32,
    /// `GuiWinGame`, selected by `S2BPacketChangeGameState` reason 4.
    win_game: Option<WinGameScreen>,
    /// A terminal action queued by the credits ticker. The driver drains this after releasing
    /// the mutable screen borrow, just as it drains other screen-owned action streams.
    pending_action: Option<Action>,
}

impl GameOverScreen {
    /// A fresh death screen with its buttons disabled, as vanilla's `initGui` leaves them.
    #[must_use]
    pub fn new(hardcore: bool, score: i32) -> Self {
        Self {
            buttons: Vec::new(),
            timer: 0,
            hardcore,
            score,
            laid_out: false,
            quit_confirmation: None,
            confirmation_message: String::new(),
            confirmation_title: String::new(),
            confirmation_respawn: String::new(),
            width: 0,
            height: 0,
            win_game: None,
            pending_action: None,
        }
    }

    /// Construct the multiplayer End-poem screen. java: `new GuiWinGame()` from
    /// `NetHandlerPlayClient.handleChangeGameState` reason 4.
    #[must_use]
    pub fn new_win_game(
        end_text: &str,
        credits_text: &str,
        username: &str,
        font: &FontRenderer,
    ) -> Self {
        let mut screen = Self::new(false, 0);
        screen.win_game = Some(WinGameScreen::from_texts(
            end_text,
            credits_text,
            username,
            font,
        ));
        screen
    }

    /// Whether this is the End-poem screen rather than the ordinary death screen.
    #[must_use]
    pub fn is_win_game(&self) -> bool {
        self.win_game.is_some()
    }

    /// Take a respawn action emitted by the end-poem ticker, at most once.
    pub fn take_action(&mut self) -> Option<Action> {
        self.pending_action.take()
    }

    /// Whether the buttons have woken up yet.
    #[must_use]
    pub fn buttons_enabled(&self) -> bool {
        if self.win_game.is_some() {
            return false;
        }
        !self.buttons.is_empty() && self.buttons.iter().all(|button| button.enabled)
    }

    /// Ticks since the player died. java: `enableButtonsTimer`
    #[must_use]
    pub fn timer(&self) -> i32 {
        self.timer
    }

    /// The score captured from `EntityPlayer.getScore()` when this screen opened.
    ///
    /// This is intentionally separate from the player's experience total; vanilla's death
    /// screen reads DataWatcher index 18 through `getScore()`, not `experienceTotal`.
    #[must_use]
    pub fn score(&self) -> i32 {
        self.score
    }

    /// Lay the buttons out. java: `initGui`
    ///
    /// Called on every resize, exactly as vanilla's is. The death view re-disables its buttons
    /// without resetting `enableButtonsTimer`; after tick 20, vanilla's equality-only wakeup has
    /// passed and a resize therefore strands those replacement buttons disabled.
    // java: GuiGameOver.java:21
    pub fn init(&mut self, ctx: &ScreenCtx, lang: &Language) {
        let (w, h) = (ctx.width, ctx.height);
        // The client driver calls the screen's draw path every frame. Java only invokes
        // `initGui` when the screen is first shown or resized; rebuilding here would disable
        // the buttons again after every frame and make `enableButtonsTimer == 20` unreachable.
        if self.laid_out && self.width == w && self.height == h {
            return;
        }
        if let Some(win_game) = &mut self.win_game {
            win_game.resize(w, h);
            self.width = w;
            self.height = h;
            self.laid_out = true;
            return;
        }
        let tr = |k: &str| lang.tr(k).to_owned();
        self.width = w;
        self.height = h;
        self.confirmation_message = tr("deathScreen.quit.confirm");
        self.confirmation_title = tr("deathScreen.titleScreen");
        self.confirmation_respawn = tr("deathScreen.respawn");
        if let Some(confirmation) = &mut self.quit_confirmation {
            confirmation.resize(w, h);
            self.laid_out = true;
            return;
        }
        self.buttons = if self.hardcore {
            // java: `isIntegratedServerRunning()` picks `deleteWorld`; the multiplayer-only rule means this
            // client never runs one, so it is always the "leave server" wording.
            vec![Button::new(
                ids::TITLE,
                w / 2 - 100,
                h / 4 + 96,
                tr("deathScreen.leaveServer"),
            )]
        } else {
            vec![
                Button::new(
                    ids::RESPAWN,
                    w / 2 - 100,
                    h / 4 + 72,
                    tr("deathScreen.respawn"),
                ),
                Button::new(
                    ids::TITLE,
                    w / 2 - 100,
                    h / 4 + 96,
                    tr("deathScreen.titleScreen"),
                ),
            ]
        };
        // java: the loop at the end of `initGui` disables every button it just added.
        for button in &mut self.buttons {
            button.enabled = false;
        }
        self.laid_out = true;
    }

    /// Whether [`GameOverScreen::init`] has run.
    #[must_use]
    pub fn is_laid_out(&self) -> bool {
        self.laid_out
    }

    fn enable_buttons(&mut self) {
        for button in &mut self.buttons {
            button.enabled = true;
        }
    }

    fn open_quit_confirmation(&mut self) {
        let mut confirmation = YesNoScreen::new(
            self.confirmation_message.clone(),
            "",
            self.confirmation_title.clone(),
            self.confirmation_respawn.clone(),
            0,
        )
        .with_world_loaded(true);
        confirmation.resize(self.width, self.height);
        // java: `guiyesno.setButtonDelay(20)` after `displayGuiScreen(guiyesno)` has laid it out.
        confirmation.set_button_delay(20);
        self.quit_confirmation = Some(confirmation);
    }

    fn close_quit_confirmation(&mut self) {
        self.quit_confirmation = None;
        self.timer = 0;
        self.buttons = vec![
            Button::new(
                ids::RESPAWN,
                self.width / 2 - 100,
                self.height / 4 + 72,
                self.confirmation_respawn.clone(),
            ),
            Button::new(
                ids::TITLE,
                self.width / 2 - 100,
                self.height / 4 + 96,
                self.confirmation_title.clone(),
            ),
        ];
        for button in &mut self.buttons {
            button.enabled = false;
        }
    }

    /// One client tick. java: `updateScreen`
    // java: GuiGameOver.java:135
    pub fn tick(&mut self) {
        if let Some(win_game) = &mut self.win_game {
            if win_game.tick() {
                self.pending_action = Some(Action::Respawn);
            }
            return;
        }
        if let Some(confirmation) = &mut self.quit_confirmation {
            Screen::tick(confirmation);
            return;
        }
        self.timer += 1;
        // java: `== 20`, not `>= 20`; this is observable after `initGui` replaces the buttons.
        if self.timer == ENABLE_BUTTONS_TICK {
            self.enable_buttons();
        }
    }

    /// Handle a click, including the embedded `GuiYesNo` quit confirmation.
    // java: GuiGameOver.java:64 (actionPerformed)
    pub fn mouse_clicked(&mut self, x: i32, y: i32, button: i32) -> Action {
        if self.win_game.is_some() {
            // java: GuiWinGame has no buttons and inherits GuiScreen.mouseClicked's no-op.
            return Action::None;
        }
        if button != 0 {
            return Action::None;
        }
        if let Some(confirmation) = &mut self.quit_confirmation {
            return match confirmation.click(x, y, button) {
                Some(result) if result.confirmed => Action::Disconnect,
                Some(_) => Action::Respawn,
                None => Action::None,
            };
        }
        let Some(id) = self
            .buttons
            .iter()
            .find(|b| b.mouse_pressed(x, y))
            .map(|b| b.id)
        else {
            return Action::None;
        };
        match id {
            ids::RESPAWN => Action::Respawn,
            ids::TITLE if self.hardcore => Action::Disconnect,
            ids::TITLE => {
                self.open_quit_confirmation();
                Action::None
            }
            _ => Action::None,
        }
    }

    /// Draw. java: `drawScreen`
    // java: GuiGameOver.java:105
    pub fn draw(
        &mut self,
        d: &mut DrawList,
        font: &mut FontRenderer,
        ctx: &ScreenCtx,
        mouse: (i32, i32),
        lang: &Language,
    ) {
        if let Some(win_game) = &mut self.win_game {
            win_game.draw(d, font, ctx, ctx.partial_ticks);
            return;
        }
        if let Some(confirmation) = &mut self.quit_confirmation {
            confirmation.draw_dialog(d, font, mouse);
            return;
        }
        d.gradient_rect(
            0,
            0,
            ctx.width,
            ctx.height,
            DEATH_GRADIENT_TOP,
            DEATH_GRADIENT_BOTTOM,
        );
        // java: `scale(2, 2, 2)` around the title, then `drawCenteredString(.., width / 2 / 2,
        // 30, ..)` — the halved x is the pre-scale coordinate, so the title lands centred.
        d.push_transform(Transform::IDENTITY.then_scale(2.0));
        let title = if self.hardcore {
            lang.tr("deathScreen.title.hardcore")
        } else {
            lang.tr("deathScreen.title")
        };
        draw_centered_string(d, font, title, ctx.width / 2 / 2, 30, TITLE_COLOR);
        d.pop_transform();

        if self.hardcore {
            draw_centered_string(
                d,
                font,
                lang.tr("deathScreen.hardcoreInfo"),
                ctx.width / 2,
                144,
                TITLE_COLOR,
            );
        }

        // java: `I18n.format("deathScreen.score") + ": " + EnumChatFormatting.YELLOW + score`
        let score = format!("{}: §e{}", lang.tr("deathScreen.score"), self.score);
        draw_centered_string(d, font, &score, ctx.width / 2, 100, TITLE_COLOR);

        for b in &mut self.buttons {
            b.draw(d, font, mouse.0, mouse.1);
        }
    }

    /// Route a key through the screen and expose the End-poem's Escape shortcut as an application
    /// action. java: `GuiWinGame.keyTyped` calls `sendRespawnPacket` for Escape.
    pub fn key_typed_action(&mut self, typed: Option<char>, key: i32, mods: Modifiers) -> Action {
        if self.win_game.is_some() {
            if key == crate::widget::keys::ESCAPE {
                return Action::Respawn;
            }
            return Action::None;
        }
        match Screen::key_typed(self, typed, key, mods) {
            ScreenAction::Pop | ScreenAction::CloseAll => Action::CloseAll,
            ScreenAction::None => Action::None,
        }
    }
}

impl Screen for GameOverScreen {
    fn name(&self) -> &'static str {
        "game_over"
    }

    /// **Vanilla swallows every key, Escape included.** `GuiGameOver.keyTyped` is an empty
    /// override, so the death screen cannot be dismissed with the keyboard — the only ways out
    /// are its two buttons. Reproduced literally, because the default
    /// [`Screen::key_typed`](crate::screen_stack::Screen::key_typed) would pop on Escape and
    /// hand the player a world they are dead in.
    // java: GuiGameOver.java:57
    fn key_typed(&mut self, _typed: Option<char>, key: i32, _mods: Modifiers) -> ScreenAction {
        if self.win_game.is_some() {
            // The driver uses `key_typed_action` so Escape can emit C16. Keep the trait-level
            // method modal for callers that drive the screen through a generic ScreenStack.
            return ScreenAction::None;
        }
        // `GuiGameOver` itself swallows Escape, but its `GuiYesNo` child inherits
        // `GuiScreen.keyTyped`: Escape calls `displayGuiScreen(null)`, which substitutes a fresh
        // disabled `GuiGameOver` while the player is still dead.
        if self.quit_confirmation.is_some() && key == crate::widget::keys::ESCAPE {
            self.close_quit_confirmation();
        }
        ScreenAction::None
    }

    /// java: `GuiGameOver.doesGuiPauseGame` returns `false`, while `GuiWinGame` returns `true`.
    // java: GuiGameOver.java:127
    fn pauses_game(&self) -> bool {
        self.win_game.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::widget::ScaledResolution;

    fn ctx() -> ScreenCtx {
        ScreenCtx::new(ScaledResolution::new(854, 480, 1, false), 0.0)
    }

    fn lang() -> Language {
        let mut lang = Language::new("en_US");
        lang.insert("deathScreen.title", "You died!");
        lang.insert("deathScreen.title.hardcore", "Game over!");
        lang.insert("deathScreen.respawn", "Respawn");
        lang.insert("deathScreen.titleScreen", "Title screen");
        lang.insert("deathScreen.leaveServer", "Leave server");
        lang.insert("deathScreen.quit.confirm", "Are you sure you want to quit?");
        lang.insert("deathScreen.score", "Score");
        lang.insert(
            "deathScreen.hardcoreInfo",
            "You cannot respawn in hardcore!",
        );
        lang
    }

    fn screen(hardcore: bool) -> GameOverScreen {
        let mut s = GameOverScreen::new(hardcore, 0);
        s.init(&ctx(), &lang());
        s
    }

    #[test]
    fn the_buttons_land_on_vanillas_coordinates() {
        let s = screen(false);
        let by = |id: i32| s.buttons.iter().find(|b| b.id == id).expect("button");
        assert_eq!(
            (by(ids::RESPAWN).rect.x, by(ids::RESPAWN).rect.y),
            (854 / 2 - 100, 480 / 4 + 72)
        );
        assert_eq!(by(ids::RESPAWN).rect.width, 200);
        assert_eq!(by(ids::TITLE).rect.y, 480 / 4 + 96);
    }

    /// The whole point of `enableButtonsTimer`: the click that killed the player must not also
    /// dismiss the screen.
    #[test]
    fn the_buttons_are_dead_for_exactly_twenty_ticks() {
        let mut s = screen(false);
        let respawn = s.buttons.iter().find(|b| b.id == ids::RESPAWN).unwrap();
        let (x, y) = (respawn.rect.x + 1, respawn.rect.y + 1);
        assert!(!s.buttons_enabled());
        for tick in 0..ENABLE_BUTTONS_TICK {
            assert_eq!(
                s.mouse_clicked(x, y, 0),
                Action::None,
                "tick {tick} accepted a click"
            );
            s.tick();
        }
        assert!(s.buttons_enabled());
        assert_eq!(s.timer(), ENABLE_BUTTONS_TICK);
        assert_eq!(s.mouse_clicked(x, y, 0), Action::Respawn);
    }

    #[test]
    fn each_button_produces_its_action() {
        let mut s = screen(false);
        for _ in 0..ENABLE_BUTTONS_TICK {
            s.tick();
        }
        let hit = |s: &mut GameOverScreen, id: i32| {
            let b = s.buttons.iter().find(|b| b.id == id).unwrap();
            let (x, y) = (b.rect.x + 1, b.rect.y + 1);
            s.mouse_clicked(x, y, 0)
        };
        assert_eq!(hit(&mut s, ids::RESPAWN), Action::Respawn);
        assert_eq!(hit(&mut s, ids::TITLE), Action::None);
        assert!(s.quit_confirmation.is_some());
        assert_eq!(s.mouse_clicked(0, 0, 0), Action::None);
        // Right-clicks are ignored, as `GuiScreen.mouseClicked` only fires on button 0.
        assert_eq!(s.mouse_clicked(854 / 2 + 6, 480 / 6 + 97, 1), Action::None);
    }

    #[test]
    fn title_screen_requires_a_second_delayed_confirmation_click() {
        let mut s = screen(false);
        for _ in 0..ENABLE_BUTTONS_TICK {
            s.tick();
        }
        let title = s.buttons.iter().find(|b| b.id == ids::TITLE).unwrap();
        let title_point = (title.rect.x + 1, title.rect.y + 1);
        assert_eq!(
            s.mouse_clicked(title_point.0, title_point.1, 0),
            Action::None
        );
        assert!(s.quit_confirmation.is_some());

        let yes = (854 / 2 - 154, 480 / 6 + 97);
        let no = (854 / 2 + 6, 480 / 6 + 97);
        for _ in 0..19 {
            assert_eq!(s.mouse_clicked(yes.0, yes.1, 0), Action::None);
            assert_eq!(s.mouse_clicked(no.0, no.1, 0), Action::None);
            s.tick();
        }
        assert_eq!(s.mouse_clicked(yes.0, yes.1, 0), Action::None);
        s.tick();
        assert_eq!(s.mouse_clicked(yes.0, yes.1, 0), Action::Disconnect);
    }

    #[test]
    fn declining_the_quit_confirmation_respawns() {
        let mut s = screen(false);
        for _ in 0..ENABLE_BUTTONS_TICK {
            s.tick();
        }
        let title = s.buttons.iter().find(|b| b.id == ids::TITLE).unwrap();
        assert_eq!(
            s.mouse_clicked(title.rect.x + 1, title.rect.y + 1, 0),
            Action::None
        );
        for _ in 0..20 {
            s.tick();
        }
        assert_eq!(
            s.mouse_clicked(854 / 2 + 6, 480 / 6 + 97, 0),
            Action::Respawn
        );
    }

    /// java: hardcore drops the Respawn button altogether, which is the difference between
    /// "press a button to carry on" and "the run is over".
    #[test]
    fn hardcore_has_no_respawn_button() {
        let mut s = screen(true);
        assert_eq!(s.buttons.len(), 1);
        assert!(s.buttons.iter().all(|b| b.id == ids::TITLE));
        for _ in 0..ENABLE_BUTTONS_TICK {
            s.tick();
        }
        let b = &s.buttons[0];
        let (x, y) = (b.rect.x + 1, b.rect.y + 1);
        assert_eq!(s.mouse_clicked(x, y, 0), Action::Disconnect);
    }

    /// java: `GuiGameOver.keyTyped` is an **empty** override, so Escape does nothing. The
    /// default `Screen::key_typed` pops, which would drop the player back into a world they are
    /// dead in with no way to respawn.
    #[test]
    fn escape_does_not_dismiss_the_death_screen() {
        let mut s = screen(false);
        assert_eq!(
            s.key_typed(None, crate::widget::keys::ESCAPE, Modifiers::NONE),
            ScreenAction::None
        );
        assert_eq!(
            s.key_typed(Some('e'), 18, Modifiers::NONE),
            ScreenAction::None
        );
    }

    /// `GuiYesNo` inherits Escape-to-null; because the player is still dead, Minecraft
    /// substitutes a fresh, disabled `GuiGameOver` rather than returning to play.
    #[test]
    fn escape_from_confirmation_substitutes_a_fresh_disabled_death_screen() {
        let mut s = screen(false);
        for _ in 0..ENABLE_BUTTONS_TICK {
            s.tick();
        }
        let title = s.buttons.iter().find(|b| b.id == ids::TITLE).unwrap();
        s.mouse_clicked(title.rect.x + 1, title.rect.y + 1, 0);
        assert!(s.quit_confirmation.is_some());

        assert_eq!(
            s.key_typed(None, crate::widget::keys::ESCAPE, Modifiers::NONE),
            ScreenAction::None
        );
        assert!(s.quit_confirmation.is_none());
        assert_eq!(s.timer(), 0);
        assert!(s.buttons.iter().all(|button| !button.enabled));
        assert!(s.buttons.iter().any(|button| button.id == ids::RESPAWN));
    }

    #[test]
    fn resizing_the_confirmation_reproduces_gui_yes_nos_early_enable_quirk() {
        let mut s = screen(false);
        for _ in 0..ENABLE_BUTTONS_TICK {
            s.tick();
        }
        let title = s.buttons.iter().find(|b| b.id == ids::TITLE).unwrap();
        s.mouse_clicked(title.rect.x + 1, title.rect.y + 1, 0);
        s.tick();
        assert!(!s.quit_confirmation.as_ref().unwrap().buttons_enabled());

        s.init(
            &ScreenCtx::new(ScaledResolution::new(1920, 1080, 1, false), 0.0),
            &lang(),
        );
        let confirmation = s.quit_confirmation.as_ref().unwrap();
        assert_eq!(confirmation.ticks_until_enable(), 19);
        assert!(confirmation.buttons_enabled());
    }

    #[test]
    fn a_resize_after_tick_twenty_reproduces_vanillas_permanent_lockout() {
        let mut s = screen(false);
        for _ in 0..ENABLE_BUTTONS_TICK {
            s.tick();
        }
        assert!(s.buttons_enabled());
        s.init(
            &ScreenCtx::new(ScaledResolution::new(1920, 1080, 1, false), 0.0),
            &lang(),
        );
        assert_eq!(s.timer(), ENABLE_BUTTONS_TICK);
        assert!(
            s.buttons.iter().all(|b| !b.enabled),
            "initGui disables replacements after the sole enable tick has passed"
        );
        s.tick();
        assert!(s.buttons.iter().all(|b| !b.enabled));
        assert!(!s.pauses_game());
    }

    /// java: `GuiWinGame.updateScreen` does not begin scrolling until the screen has a real
    /// viewport, then requests exactly one terminal respawn after the computed end tick.
    #[test]
    fn win_game_scroll_waits_for_layout_and_emits_one_terminal_action() {
        let mut s = WinGameScreen::new(vec!["The End".to_owned()]);
        assert!(
            !s.tick(),
            "a screen not laid out by setWorldAndResolution cannot finish"
        );
        assert_eq!(s.scroll_ticks(), 0);

        s.resize(854, 480);
        let end_tick = ((s.content_height() + 480 * 2 + 24) as f32 / WIN_GAME_SCROLL_SPEED) as i32;
        for _ in 0..end_tick {
            assert!(
                !s.tick(),
                "GuiWinGame finishes only after the strict greater-than check"
            );
        }
        assert!(s.tick());
        assert!(s.sent_respawn());
        assert!(!s.tick(), "the C16 producer must be one-shot");
    }

    /// The application wrapper owns the action queue because the driver cannot borrow a screen
    /// mutably and send through the session at the same time. Its completion must still expose
    /// one `Action::Respawn`, with no repeat on the following tick/drain.
    #[test]
    fn game_over_wrapper_queues_the_auto_respawn_action_once() {
        let mut s = GameOverScreen::new(false, 0);
        s.win_game = Some(WinGameScreen::new(vec!["The End".to_owned()]));
        s.init(&ctx(), &lang());
        let end_tick = ((s.win_game.as_ref().unwrap().content_height() + 480 * 2 + 24) as f32
            / WIN_GAME_SCROLL_SPEED) as i32;
        for _ in 0..end_tick {
            s.tick();
            assert!(s.take_action().is_none());
        }
        s.tick();
        assert_eq!(s.take_action(), Some(Action::Respawn));
        assert!(s.take_action().is_none());
    }

    /// java: `GuiWinGame` pauses the game and consumes Escape by sending the same respawn action
    /// as the auto-scroll terminal path; the generic `Screen` boundary stays modal for callers
    /// that do not use the client driver's action adapter.
    #[test]
    fn win_game_escape_is_respawn_without_popping_the_screen() {
        let mut s = GameOverScreen::new(false, 0);
        s.win_game = Some(WinGameScreen::new(Vec::new()));
        assert!(s.is_win_game());
        assert!(s.pauses_game());
        assert_eq!(
            s.key_typed_action(None, crate::widget::keys::ESCAPE, Modifiers::NONE),
            Action::Respawn
        );
        assert_eq!(
            s.key_typed(None, crate::widget::keys::ESCAPE, Modifiers::NONE),
            ScreenAction::None
        );
    }
}
