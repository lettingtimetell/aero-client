//! The client half: input, interaction and packets.
//!
//! * **Modern Keybinds** (`modern_keybinds`, Controls, default ON, item 4). 1.8.9 releases every
//!   binding when a screen opens (`KeyBinding.unPressAllKeys`) and never looks at the keyboard
//!   again when it closes, so a W held through the inventory does nothing until it is pressed
//!   again. Vanilla since 1.13 (`KeyMapping.setAll`) re-reads the keys that are still down.
//!   [`Client::x4_repress_held_keys`] does that for the bindings vanilla
//!   polls with `isKeyDown()` every tick ([`HELD_BINDINGS`]) at the first tick back in game,
//!   through [`mc_render::KeyBindings::repress_if_held`], which sets the held flag only: the
//!   press counter `isPressed()` reads stays empty, so the inventory, drop, chat, hotbar and
//!   perspective keys can never fire from it. The mouse buttons are deliberately not in the set:
//!   a right button still held as a chest closes would reopen the chest every four ticks.
//!   Hook: In `driver.rs` `handle_game_input`.
//! * **Binding F4, F2 and F11** (item 7). `key_pressed` handled the raw F4 shader toggle and the
//!   screenshot/fullscreen bindings before any screen, so the controls screen (and the mod
//!   menu's keybind capture) never received those keys. F4 is now in-game only, as in vanilla,
//!   and a screen waiting for a key ([`Client::x4_key_capture_active`]) gets it first. Hooks:
//!   In `driver.rs` `key_pressed`.
//! * **Sound-slider saves** (item 16). `GuiScreenOptionsSounds.Button.mouseDragged` calls
//!   `saveOptions` on every drawn frame of a volume drag; mc-rs reproduced it, so a held slider
//!   rewrote `options.txt` (and sent a C15) every frame. The C15 is now de-duplicated in
//!   [`crate::session::Session::send_client_settings_if_changed`]; the file write of a held
//!   slider is postponed to its release ([`Client::x4_defer_sound_save`]). Hooks:
//!   In `drain_sound_options_events` and `show_screen`.
//! * **Adventure-mode outlines** (item 18). [`Client::x4_draws_block_outline`] ports
//!   `EntityRenderer.isDrawBlockOutline`, plus Block Overlay's "Show in adventure mode". Hook:
//!   In `driver.rs` `selection_box`.
//! * **Using items stops mining** (formerly "Eating stops mining"). 1.7 Animations'
//!   `use_stops_mining`:
//!   [`Client::x4_use_stops_mining`]. Hook: In `driver.rs` `right_click_mouse`.
//!
//! Elsewhere (not in this file): item 6 renews held-stack identities in the container click
//! machine (`mc-ui` `container/model.rs`), item 17 draws the torch box
//! (`mc-world` `collision::torch_bounds`), item 3 makes the pick's
//! candidate reject exact (`interact::box_within_reach`).
//!
//! # Cost
//!
//! In game, one tick costs two `bool` tests (plus one settings-generation compare on the tick
//! after a screen closed). Per frame, the outline asks for the adventure option (one generation
//! compare) and, only when the player lacks `allowEdit`, reads the held item's NBT lists.

use mc_render::input::keys;
use mc_render::{InputState, KeyBindings};

use super::super::{Client, Ui};

const MODERN_KEYBINDS: &str = "modern_keybinds";
const BLOCK_OVERLAY: &str = "block_overlay";
const ADVENTURE_OUTLINE: &str = "adventure_outline";
/// 1.7 Animations and its "Using items stops mining" option, which replaced Modern Keybinds' `eat_stops_mining`.
const OLD_ANIMATIONS: &str = "old_animations";
const USE_STOPS_MINING: &str = "use_stops_mining";
/// The ticks a dig must have damaged its block (attack held for
/// 150 ms) before a use press may end it. Mining is a held button; a PvP click that lands on a
/// block starts a dig too, but one released within a tick or two. Block-hitting near blocks
/// therefore stays exactly vanilla (the press is dropped, as every 1.8.9 client does), and the
/// one non-vanilla packet timing is limited to real mining. A use held from the start of a dig
/// still takes over three ticks later, through vanilla's held-button repeat.
const USE_STOPS_MINING_MIN_DIG_TICKS: u32 = 3;

/// The bindings Modern Keybinds re-applies: every one vanilla reads with `isKeyDown()` each tick
/// (`MovementInputFromOptions.updatePlayerMoveState`, `EntityPlayerSP.onLivingUpdate`'s sprint
/// checks, `GuiIngame`'s player list, the spectator outline key). Edge bindings (`isPressed()`:
/// inventory, drop, chat, command, hotbar, perspective, smooth camera, pick block) are absent,
/// and could not fire anyway because re-pressing queues no press.
pub(crate) const HELD_BINDINGS: [&str; 9] = [
    keys::FORWARD,
    keys::LEFT,
    keys::BACK,
    keys::RIGHT,
    keys::JUMP,
    keys::SNEAK,
    keys::SPRINT,
    keys::PLAYER_LIST,
    keys::SPECTATOR_OUTLINES,
];

/// The client state: a settings snapshot plus two latches.
#[derive(Debug, Default)]
pub(crate) struct X4State {
    /// The settings generation last read; `0` = never (the store starts at 1).
    seen: u64,
    /// `modern_keybinds` is on.
    modern_keybinds: bool,
    /// The game has not owned the bindings since the last re-press: a screen was open at a tick,
    /// or a screen-open unpress was drained. The first tick back in game re-applies held keys.
    repress_armed: bool,
    /// A held volume slider skipped its `options.txt` write; the release (or leaving the screen)
    /// writes it.
    sound_save_pending: bool,
    /// Block Overlay is on with "Show in adventure mode".
    adventure_outline: bool,
}

impl X4State {
    fn sync(&mut self, settings: &mc_mods::ModSettings) {
        if self.seen == settings.generation() {
            return;
        }
        self.seen = settings.generation();
        self.modern_keybinds = settings.enabled(MODERN_KEYBINDS);
        self.adventure_outline =
            settings.enabled(BLOCK_OVERLAY) && settings.bool(BLOCK_OVERLAY, ADVENTURE_OUTLINE);
    }
}

/// The blocks whose tile entity is an `IInventory` (`TileEntityLockable`'s subclasses): the only
/// ones `isDrawBlockOutline` outlines for a spectator. The client creates a block's tile entity
/// on first access, so `getTileEntity(pos) instanceof IInventory` is a property of the block.
fn is_inventory_block(state: mc_data::BlockState) -> bool {
    matches!(
        mc_data::block_info(state.block())
            .java_class
            .rsplit('.')
            .next(),
        Some(
            "BlockChest"
                | "BlockFurnace"
                | "BlockDispenser"
                | "BlockDropper"
                | "BlockHopper"
                | "BlockBrewingStand"
                | "BlockBeacon"
        )
    )
}

impl Client {
    /// Modern Keybinds, at the start of `handle_game_input`, right after the
    /// pending screen-open unpress was drained. `had_unpress` says whether one was pending: a
    /// screen that opened *and* closed between two ticks leaves no non-game tick behind, only
    /// that unpress.
    pub(crate) fn x4_repress_held_keys(
        &mut self,
        had_unpress: bool,
        input: &InputState,
        keys: &mut KeyBindings,
    ) {
        let in_game = matches!(self.ui, Ui::InGame);
        if had_unpress || !in_game {
            self.x4.repress_armed = true;
        }
        if !in_game || !self.x4.repress_armed {
            return;
        }
        self.x4.repress_armed = false;
        self.x4.sync(&self.mods.settings);
        if !self.x4.modern_keybinds {
            return;
        }
        for name in HELD_BINDINGS {
            keys.repress_if_held(name, |bound| input.is_down(bound));
        }
    }

    /// Asked before each `SoundOptionsEvent::SaveSettings`: while a volume
    /// slider is held the `options.txt` write waits for the release. The live volume is applied
    /// by the event's `ApplyVolume` either way, and the C15 that `saveOptions` would send is a
    /// duplicate (see `Session::send_client_settings_if_changed`).
    pub(crate) fn x4_defer_sound_save(&mut self) -> bool {
        let held = self.x4_sound_slider_held();
        if held {
            self.x4.sound_save_pending = true;
        }
        held
    }

    /// After a sound-options input pass: write the postponed save once no
    /// slider is held any more. `leaving` (the screen is being replaced) writes it regardless.
    pub(crate) fn x4_flush_sound_save(&mut self, leaving: bool) {
        if !self.x4.sound_save_pending || (!leaving && self.x4_sound_slider_held()) {
            return;
        }
        self.x4.sound_save_pending = false;
        self.save_settings_quietly();
    }

    fn x4_sound_slider_held(&self) -> bool {
        matches!(&self.ui, Ui::SoundOptions(screen)
            if screen.sliders.iter().any(|(_, slider)| slider.dragging))
    }

    /// Block Overlay's "Show in adventure mode" (Block Overlay on and the
    /// option set), from the settings snapshot: one generation compare per call.
    pub(crate) fn x4_adventure_outline(&mut self) -> bool {
        self.x4.sync(&self.mods.settings);
        self.x4.adventure_outline
    }

    /// java: `EntityRenderer.isDrawBlockOutline()` for a block hit, minus the
    /// `hideGUI` test the caller made first (item 18):
    ///
    /// ```text
    /// flag = entity instanceof EntityPlayer && !hideGUI;
    /// if (flag && !((EntityPlayer) entity).capabilities.allowEdit) {
    ///     if (playerController.getCurrentGameType() == SPECTATOR)
    ///         flag = block.hasTileEntity() && getTileEntity(pos) instanceof IInventory;
    ///     else
    ///         flag = itemstack != null && (itemstack.canDestroy(block) || itemstack.canPlaceOn(block));
    /// }
    /// ```
    ///
    /// `entity` is the render-view entity: spectating a mob outlines nothing, and a spectated
    /// player is an `EntityOtherPlayerMP` whose `capabilities.allowEdit` is the `true` default.
    /// `adventure_outline` (Block Overlay's option) outlines every block in adventure mode.
    pub(crate) fn x4_draws_block_outline(
        session: &crate::session::Session,
        state: mc_data::BlockState,
        held: Option<&mc_protocol::ItemStack>,
        adventure_outline: bool,
    ) -> bool {
        match session.render_view_target() {
            crate::session::RenderViewTarget::Entity { network_id, .. } => session
                .state()
                .entities
                .get(network_id)
                .is_some_and(|tracked| tracked.kind == crate::sync::TrackedKind::Player),
            crate::session::RenderViewTarget::Local => {
                let local = &session.state().local;
                if local.abilities.allow_edit {
                    true
                } else if local.game_mode == mc_protocol::play::s2c::GameMode::Spectator {
                    is_inventory_block(state)
                } else {
                    adventure_outline
                        || crate::app::interact::held_can_destroy(held, state)
                        || crate::app::interact::held_can_place_on(held, state)
                }
            }
        }
    }

    /// A screen is waiting for a key to bind: the controls screen's armed row or
    /// the mod menu's keybind capture. `key_pressed` then gives it the key ahead of the
    /// screenshot and fullscreen bindings (item 7).
    pub(crate) fn x4_key_capture_active(&self) -> bool {
        match &self.ui {
            Ui::Controls(screen) => screen.arming.is_some(),
            Ui::ModMenu(screen) => screen.capturing_key(),
            _ => false,
        }
    }

    /// 1.7 Animations' "Using items stops mining", which extends
    /// Modern Keybinds' earlier "Eating stops mining"to sword
    /// blocking and bow drawing. Asked
    /// by `right_click_mouse` only while a block is being broken, where vanilla's
    /// `if (!playerController.getIsHittingBlock())` drops the click.
    ///
    /// When 1.7 Animations and the option are on, the dig has run for
    /// [`USE_STOPS_MINING_MIN_DIG_TICKS`] ticks (a held button, not a PvP click), and the press
    /// would really start a use ([`crate::app::interact::right_click_starts_use`]: the held item's
    /// own guard passes, and the click is not taken by the block under the crosshair), the dig
    /// is cancelled exactly as `PlayerControllerMP.resetBlockRemoving` does when attack is
    /// released (C07 `ABORT_DESTROY_BLOCK` at `currentBlock` facing down, `isHittingBlock` false,
    /// the crack cleared) and `true` tells `right_click_mouse` to go on as vanilla's would one
    /// tick later.
    /// The packets are therefore the ones of "release attack, then right-click" (targeted C08 at
    /// the block, untargeted C08), minus the tick between them: no extra swing, C0A, START or STOP.
    /// The use then holds the dig off through vanilla's own `!isUsingItem()` gate. When it ends
    /// with attack still held, vanilla's controller carries on as after any reset on the same
    /// block: `isHittingPosition` ignores `isHittingBlock`, so the dig resumes from zero damage
    /// without a new START (a fresh attack press sends one), and with `isHittingBlock` false the
    /// next use press is vanilla's own (no ABORT at all). Anything else returns `false` and
    /// changes nothing. Cost: nothing outside a use press during a dig.
    pub(crate) fn x4_use_stops_mining(&mut self) -> bool {
        use crate::app::interact::{HitResult, UseAim};
        use mc_protocol::play::s2c::GameMode;
        let settings = &self.mods.settings;
        if !settings.enabled(OLD_ANIMATIONS)
            || !settings.bool(OLD_ANIMATIONS, USE_STOPS_MINING)
            || self.controller.dig_ticks() < USE_STOPS_MINING_MIN_DIG_TICKS
        {
            return false;
        }
        let starts = {
            let (Some(held), Some(session)) = (self.held_item(), self.session.as_ref()) else {
                return false;
            };
            let state = session.state();
            let aim = match self.hit {
                HitResult::Miss => UseAim::Nothing,
                HitResult::Entity(_) => UseAim::Entity,
                // java: `if (material != air)`: an air block goes straight to `sendUseItem`.
                HitResult::Block(hit) if state.world.block_at(hit.pos).props().is_air() => {
                    UseAim::Nothing
                }
                HitResult::Block(hit) => UseAim::Block {
                    consumed: crate::app::interact::block_right_click_consumes(
                        &state.world,
                        hit.pos,
                        hit.face,
                        held,
                        session
                            .predictor()
                            .is_some_and(|predictor| predictor.state.sneaking),
                        state.local.abilities.allow_edit,
                        state.local.game_mode == GameMode::Creative,
                    ),
                },
            };
            // A spectator never digs, and its `sendUseItem` returns false.
            state.local.game_mode != GameMode::Spectator
                && crate::app::interact::right_click_starts_use(held, self.use_guards(), aim)
        };
        if !starts {
            return false;
        }
        let step = self.controller.reset_block_removing();
        self.apply_dig_step(step);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mc_render::input::Input;
    use winit::keyboard::KeyCode;

    /// A client over the real assets in a temp game dir. Skips without the vanilla jar (set
    /// `MC_JARS_DIR`).
    fn offline_client() -> Option<(Client, tempfile::TempDir)> {
        let dir = tempfile::tempdir().expect("tempdir");
        let config = crate::app::config::ClientConfig {
            game_dir: dir.path().to_path_buf(),
            login: crate::app::config::LoginMode::Offline("Dev".to_owned()),
            ..crate::app::config::ClientConfig::default()
        }
        .with_discovered_assets();
        if !config.version_jar.as_ref().is_some_and(|p| p.is_file()) {
            eprintln!("skip: no vanilla jar in this checkout");
            return None;
        }
        let client = Client::new(config, &mut |_, _| {}).expect("the real assets must load");
        Some((client, dir))
    }

    /// The receiving end of a test session's outbound queue.
    pub(super) type ToNet = tokio::sync::mpsc::UnboundedReceiver<crate::net::ToNet>;

    /// [`offline_client`] holding a joined, socket-less session (so closing a screen returns to
    /// the game, not the title screen), with the join-time packets already drained.
    pub(super) fn client_in_game() -> Option<(
        Client,
        tempfile::TempDir,
        ToNet,
        tokio::sync::mpsc::Sender<crate::net::ToGame>,
    )> {
        let (mut client, dir) = offline_client()?;
        let (handle, to_game, mut to_net) = crate::net::channel::NetHandle::new_pair();
        let mut session = crate::session::Session::new(handle, client.join_settings());
        to_game
            .try_send(crate::net::ToGame::Packet(
                mc_protocol::play::S2CPlay::JoinGame(mc_protocol::play::s2c::JoinGame {
                    entity_id: 1,
                    hardcore_mode: false,
                    game_mode: 0,
                    dimension: 0,
                    difficulty: 2,
                    max_players: 20,
                    level_type: "default".into(),
                    reduced_debug_info: false,
                }),
            ))
            .expect("channel");
        session.poll(16);
        client.session = Some(session);
        client.ui = Ui::InGame;
        let _ = drained(&mut to_net);
        Some((client, dir, to_net, to_game))
    }

    /// Everything the client put on the wire since the last call, in order.
    pub(super) fn drained(to_net: &mut ToNet) -> Vec<mc_protocol::play::C2SPlay> {
        let mut out = Vec::new();
        while let Ok(message) = to_net.try_recv() {
            if let crate::net::ToNet::Send(packet) = message {
                out.push(packet);
            }
        }
        out
    }

    fn held(raw: &mut InputState, keys: &mut KeyBindings, code: KeyCode) {
        raw.on_key(code, true, false);
        keys.set_state(Input::Key(code), true);
    }

    /// Item 4, end to end through `handle_game_input`: W and Ctrl held while the inventory
    /// opens and closes walk on at the first tick back in game; E, Q and 1 held across the same
    /// close fire nothing. Off, the same sequence is vanilla: nothing moves until W is pressed
    /// again.
    #[test]
    fn held_movement_keys_resume_after_a_screen_closes_and_edge_keys_never_fire() {
        let Some((mut client, _dir, _to_net, _to_game)) = client_in_game() else {
            return;
        };
        assert!(
            client.mods.settings.enabled(MODERN_KEYBINDS),
            "Modern Keybinds is on by default"
        );
        for modern in [true, false] {
            client.mods.settings.set_enabled(MODERN_KEYBINDS, modern);
            client.ui = Ui::InGame;
            let mut keys = client.settings.keys.clone();
            let mut raw = InputState::new();
            held(&mut raw, &mut keys, KeyCode::KeyW);
            held(&mut raw, &mut keys, KeyCode::ControlLeft);
            client.handle_game_input(&raw, &mut keys);
            assert!(keys.is_down(keys::FORWARD), "control: W walks in game");

            // A screen opens (any screen: the inventory, chat, pause): vanilla unpresses.
            client.show_screen(Some(Ui::Pause(Box::new(
                mc_ui::screens::pause::PauseScreen::new(),
            ))));
            client.handle_game_input(&raw, &mut keys);
            assert!(!keys.is_down(keys::FORWARD) && !keys.is_down(keys::SPRINT));
            // Keys pressed while the screen is open never reach the bindings (the shell gates
            // edges on `game_input_allowed`), but the keyboard still holds them.
            raw.on_key(KeyCode::KeyE, true, false);
            raw.on_key(KeyCode::KeyQ, true, false);
            raw.on_key(KeyCode::Digit1, true, false);
            client.handle_game_input(&raw, &mut keys);

            client.show_screen(None);
            assert!(matches!(client.ui, Ui::InGame));
            let slot_before = client.player_hud.selected_slot;
            client.handle_game_input(&raw, &mut keys);
            assert_eq!(
                keys.is_down(keys::FORWARD),
                modern,
                "modern={modern}: forward"
            );
            assert_eq!(
                keys.is_down(keys::SPRINT),
                modern,
                "modern={modern}: sprint"
            );
            let moved = crate::app::input::move_input(&keys, 0.0, 0.0, false);
            assert_eq!(moved.forward && moved.sprint, modern);
            // Edge bindings: nothing queued, nothing opened, no hotbar change.
            assert!(matches!(client.ui, Ui::InGame), "E opened nothing");
            assert_eq!(
                client.player_hud.selected_slot, slot_before,
                "1 changed no slot"
            );
            for name in [keys::INVENTORY, keys::DROP, &keys::hotbar(1)] {
                assert!(!keys.take_pressed(name), "{name} must not fire");
            }
            raw.on_key(KeyCode::KeyE, false, false);
            raw.on_key(KeyCode::KeyQ, false, false);
            raw.on_key(KeyCode::Digit1, false, false);
        }
    }

    /// Modern Keybinds with Toggle Sprint: a key still held as a screen
    /// closes is re-applied as held, and Toggle Sprint took that for a tap. Shift held for a
    /// shift-click while the inventory closed left the player sneaking; Ctrl held for Ctrl+Q
    /// flipped the sprint toggle. Now a key held through a screen is not a tap, with Modern
    /// Keybinds on or off, and the next real tap still toggles.
    #[test]
    fn toggle_sprint_ignores_a_key_held_through_a_screen() {
        let Some((mut client, _dir, _to_net, _to_game)) = client_in_game() else {
            return;
        };
        client.mods.settings.set_enabled("toggle_sprint", true);
        client
            .mods
            .settings
            .set_bool("toggle_sprint", "toggle_sneak", true);
        for modern in [true, false] {
            client.mods.settings.set_enabled(MODERN_KEYBINDS, modern);
            client.ui = Ui::InGame;
            let mut keys = client.settings.keys.clone();
            let mut raw = InputState::new();
            client.handle_game_input(&raw, &mut keys);
            client.m3_shell_frame(&mut raw, &keys, 0.004);
            assert!(!client.m3.toggle.sprint_toggled && !client.m3.toggle.sneak_toggled);

            // A screen opens; Shift and Ctrl go down inside it (a shift-click, Ctrl+Q).
            client.show_screen(Some(Ui::Pause(Box::new(
                mc_ui::screens::pause::PauseScreen::new(),
            ))));
            client.handle_game_input(&raw, &mut keys);
            raw.on_key(KeyCode::ShiftLeft, true, false);
            raw.on_key(KeyCode::ControlLeft, true, false);
            client.m3_shell_frame(&mut raw, &keys, 0.004);

            // It closes with both keys still down: the next tick re-applies them (Modern
            // Keybinds), and the frames after it sample them.
            client.show_screen(None);
            assert!(matches!(client.ui, Ui::InGame));
            client.m3_shell_frame(&mut raw, &keys, 0.004);
            client.handle_game_input(&raw, &mut keys);
            assert_eq!(
                keys.is_down(keys::SNEAK) && keys.is_down(keys::SPRINT),
                modern,
                "control: modern={modern} re-applies the held keys"
            );
            client.m3_shell_frame(&mut raw, &keys, 0.004);
            client.m3_shell_frame(&mut raw, &keys, 0.004);
            assert!(
                !client.m3.toggle.sneak_toggled,
                "modern={modern}: a held Shift is not a sneak tap"
            );
            assert!(
                !client.m3.toggle.sprint_toggled,
                "modern={modern}: a held Ctrl is not a sprint tap"
            );

            // Let go, then tap both in game: they toggle on, as before.
            for code in [KeyCode::ShiftLeft, KeyCode::ControlLeft] {
                raw.on_key(code, false, false);
                keys.set_state(Input::Key(code), false);
            }
            client.m3_shell_frame(&mut raw, &keys, 0.004);
            for code in [KeyCode::ShiftLeft, KeyCode::ControlLeft] {
                held(&mut raw, &mut keys, code);
            }
            client.m3_shell_frame(&mut raw, &keys, 0.004);
            assert!(
                client.m3.toggle.sneak_toggled && client.m3.toggle.sprint_toggled,
                "modern={modern}: a real tap toggles"
            );
            // Tap both again to toggle off for the next round.
            for code in [KeyCode::ShiftLeft, KeyCode::ControlLeft] {
                raw.on_key(code, false, false);
                keys.set_state(Input::Key(code), false);
            }
            client.m3_shell_frame(&mut raw, &keys, 0.004);
            for code in [KeyCode::ShiftLeft, KeyCode::ControlLeft] {
                held(&mut raw, &mut keys, code);
            }
            client.m3_shell_frame(&mut raw, &keys, 0.004);
            assert!(!client.m3.toggle.sneak_toggled && !client.m3.toggle.sprint_toggled);
        }
    }

    /// Press and release one physical key through the shell's raw path, then run the frame's
    /// screen input pass that dispatches it.
    fn tap(client: &mut Client, input: &mut InputState, code: KeyCode) {
        client.handle_raw_key(code, true, false, None);
        client.handle_raw_key(code, false, false, None);
        client.handle_screen_input(input, 854, 480);
    }

    /// Open Options > Controls and arm `description`'s rebind button.
    fn arm_controls_row(client: &mut Client, description: &str) {
        client.handle_action(mc_ui::screens::Action::OpenOptions);
        client.handle_action(mc_ui::screens::Action::OpenOptionsPage(
            mc_ui::screens::options::OptionsPage::Controls,
        ));
        let Ui::Controls(screen) = &mut client.ui else {
            panic!("the controls screen is open, not {}", client.ui.name());
        };
        screen.arming = Some(description.to_owned());
    }

    /// Item 7: F4 binds on the controls screen (it used to be eaten by the shader toggle before
    /// any screen saw it), and so do the screenshot and fullscreen keys F2 and F11, without
    /// taking a screenshot or switching fullscreen. In game, F4 still flips the shader bit.
    #[test]
    fn f4_f2_and_f11_bind_on_the_controls_screen_and_f4_still_toggles_shaders_in_game() {
        let Some((mut client, _dir, _to_net, _to_game)) = client_in_game() else {
            return;
        };
        let mut input = InputState::new();
        let fullscreen = client.settings.fullscreen;
        let shader_visible = client.shader_visible;
        for (description, code) in [
            (keys::TOGGLE_PERSPECTIVE, KeyCode::F4),
            (keys::SMOOTH_CAMERA, KeyCode::F2),
            (keys::SPECTATOR_OUTLINES, KeyCode::F11),
        ] {
            arm_controls_row(&mut client, description);
            tap(&mut client, &mut input, code);
            assert_eq!(
                client.settings.keys.get(description).map(|b| b.binding()),
                Some(Input::Key(code)),
                "{code:?} binds {description}"
            );
            let Ui::Controls(screen) = &client.ui else {
                panic!("binding keeps the controls screen open");
            };
            assert!(screen.arming.is_none(), "the row disarmed");
            assert!(!client.screenshot_requested, "{code:?}: no screenshot");
            assert_eq!(
                client.settings.fullscreen, fullscreen,
                "{code:?}: no fullscreen"
            );
            assert_eq!(
                client.shader_visible, shader_visible,
                "{code:?}: shader untouched"
            );
            client.handle_action(mc_ui::screens::Action::CloseAll);
        }
        // Control: with no row armed, F2 on the controls screen is still the screenshot key.
        client.handle_action(mc_ui::screens::Action::OpenOptions);
        client.handle_action(mc_ui::screens::Action::OpenOptionsPage(
            mc_ui::screens::options::OptionsPage::Controls,
        ));
        tap(&mut client, &mut input, KeyCode::F2);
        assert!(client.screenshot_requested, "unarmed, F2 still screenshots");
        client.screenshot_requested = false;
        client.handle_action(mc_ui::screens::Action::CloseAll);
        assert!(matches!(client.ui, Ui::InGame));
        // In game, F4 is vanilla's raw shader toggle; with no renderer (headless) vanilla's
        // `entityRenderer != null` guard keeps it inert, so only check it reaches the match.
        tap(&mut client, &mut input, KeyCode::F4);
        assert_eq!(client.shader_visible, shader_visible, "no renderer: inert");
    }

    /// Item 7, the whole key table: every LWJGL code the client maps to a winit key binds on
    /// the controls screen, survives `options.txt` and keeps LWJGL's own name: F1-F19 and the
    /// numpad included.
    #[test]
    fn every_mapped_lwjgl_key_binds_saves_and_reloads_with_its_lwjgl_name() {
        let Some((mut client, dir, _to_net, _to_game)) = client_in_game() else {
            return;
        };
        let mut input = InputState::new();
        let mut checked = 0;
        for code in 1..=255 {
            let Some(key) = mc_render::input::keycode_of_lwjgl(code) else {
                continue;
            };
            // Escape unbinds while armed (vanilla); Tab/Enter etc. are ordinary keys here.
            if key == KeyCode::Escape {
                continue;
            }
            arm_controls_row(&mut client, keys::TOGGLE_PERSPECTIVE);
            tap(&mut client, &mut input, key);
            let bound = client
                .settings
                .keys
                .get(keys::TOGGLE_PERSPECTIVE)
                .map(|b| b.binding())
                .expect("the binding exists");
            assert_eq!(bound, Input::Key(key), "lwjgl {code} ({key:?})");
            assert_eq!(bound.to_lwjgl(), code, "{key:?} writes {code}");
            client.handle_action(mc_ui::screens::Action::CloseAll);
            client.screenshot_requested = false;
            checked += 1;
        }
        assert!(checked > 100, "{checked}");
        for (code, name) in [
            (59, "F1"),
            (62, "F4"),
            (88, "F12"),
            (100, "F13"),
            (105, "F18"),
            (113, "F19"),
            (71, "NUMPAD7"),
            (82, "NUMPAD0"),
            (83, "DECIMAL"),
            (55, "MULTIPLY"),
            (74, "SUBTRACT"),
            (78, "ADD"),
            (181, "DIVIDE"),
            (156, "NUMPADENTER"),
            (141, "NUMPADEQUALS"),
            (69, "NUMLOCK"),
        ] {
            let input = Input::from_lwjgl(code);
            assert!(matches!(input, Input::Key(_)), "{code} maps to a winit key");
            assert_eq!(input.display_name(), name, "lwjgl {code}");
        }
        // options.txt round trip of the user's binding.
        client
            .settings
            .keys
            .set_binding(keys::TOGGLE_PERSPECTIVE, Input::Key(KeyCode::F4));
        let path = dir.path().join("x4-options.txt");
        client.settings.save(&path).expect("save options");
        let text = std::fs::read_to_string(&path).expect("read options");
        assert!(text.contains("key_key.togglePerspective:62"), "{text}");
        let loaded = mc_render::settings::GameSettings::load(&path).expect("load options");
        assert_eq!(
            loaded
                .keys
                .get(keys::TOGGLE_PERSPECTIVE)
                .map(|b| b.binding()),
            Some(Input::Key(KeyCode::F4))
        );
    }

    /// Item 16: dragging a volume slider for many frames sends no C15 and writes options.txt
    /// once, at the release, with the final volume. Changing the render distance still sends
    /// exactly vanilla's payload, once, however many saves follow.
    #[test]
    fn a_sound_drag_sends_no_c15_and_a_render_distance_change_sends_one() {
        use mc_protocol::play::C2SPlay;
        use mc_ui::screens::options::OptionsPage;
        use mc_ui::screens::Action;
        let Some((mut client, dir, mut to_net, _to_game)) = client_in_game() else {
            return;
        };
        let options = client.config.options_path();
        assert!(options.starts_with(dir.path()), "a temp game dir");
        let c15s = |sent: &[C2SPlay]| {
            sent.iter()
                .filter_map(|packet| match packet {
                    C2SPlay::ClientSettings(inner) => Some(inner.clone()),
                    _ => None,
                })
                .collect::<Vec<_>>()
        };

        // Pause > Options > Sounds, as a player gets there.
        client.show_screen(Some(Ui::Pause(Box::new(
            mc_ui::screens::pause::PauseScreen::new(),
        ))));
        client.handle_action(Action::OpenOptions);
        client.handle_action(Action::OpenOptionsPage(OptionsPage::Sound));
        let resolution = client.resolution(854, 480);
        let ctx = mc_ui::screen_stack::ScreenCtx::new(resolution, 0.0);
        let language = client.assets.language.clone();
        let rect = match &mut client.ui {
            Ui::SoundOptions(screen) => {
                screen.init(&ctx, &client.settings, &language);
                screen.sliders[0].1.button.rect
            }
            _ => panic!("the sound screen is open"),
        };
        let to_window = |gx: i32, gy: i32| {
            (
                f64::from(gx) * 854.0 / f64::from(resolution.scaled_width()) + 0.5,
                f64::from(gy) * 480.0 / f64::from(resolution.scaled_height()) + 0.5,
            )
        };
        let x_at = |fraction: f32| rect.x + 4 + ((rect.width - 8) as f32 * fraction) as i32;
        let _ = std::fs::remove_file(&options);
        let mut input = InputState::new();
        let (px, py) = to_window(x_at(0.1), rect.y + 5);
        input.on_cursor_moved(px, py);
        input.on_mouse_button(winit::event::MouseButton::Left, true);
        client.handle_screen_input(&mut input, 854, 480);
        // Sixty drawn frames of drag, the pointer moving on most of them: vanilla saves (and
        // sends a C15) on every one.
        for frame in 0..60 {
            let fraction = 0.1 + 0.8 * (frame as f32 / 59.0);
            let (px, py) = to_window(x_at(fraction), rect.y + 5);
            input.on_cursor_moved(px, py);
            client.handle_screen_input(&mut input, 854, 480);
            assert!(
                !options.exists(),
                "frame {frame}: no options.txt write while held"
            );
        }
        // Reproduce "move a slider, then wait": a grabbed slider still receives drag
        // callbacks each frame even without cursor movement. It must not flood C15 or disk.
        for _ in 0..240 {
            client.handle_screen_input(&mut input, 854, 480);
            assert!(!options.exists(), "a stationary hold still defers the save");
        }
        assert!(
            c15s(&drained(&mut to_net)).is_empty(),
            "no packets while held"
        );
        input.on_mouse_button(winit::event::MouseButton::Left, false);
        client.handle_screen_input(&mut input, 854, 480);
        for _ in 0..240 {
            client.handle_screen_input(&mut input, 854, 480);
        }
        let level = client.settings.sound_level("master");
        assert!((level - 0.9).abs() < 0.02, "{level}");
        let written = mc_render::settings::GameSettings::load(&options)
            .expect("the release writes options.txt");
        assert!(
            (written.sound_level("master") - level).abs() < 1e-6,
            "the final volume is saved: {} vs {level}",
            written.sound_level("master")
        );
        // Done goes back to Options (a saveOptions in vanilla) and Done again to the pause
        // screen (another one).
        client.handle_action(Action::Back);
        assert!(matches!(client.ui, Ui::Options(_)), "{}", client.ui.name());
        client.handle_action(Action::Back);
        assert!(matches!(client.ui, Ui::Pause(_)), "{}", client.ui.name());
        let sent = drained(&mut to_net);
        assert!(
            c15s(&sent).is_empty(),
            "a sound drag and two Done saves send no C15: {sent:?}"
        );

        // Render distance 12 -> 7, then the two Done saves: one C15, vanilla's payload.
        client.settings.render_distance_chunks = 7;
        client.handle_action(Action::SaveSettings);
        client.handle_action(Action::SaveSettings);
        let sent = c15s(&drained(&mut to_net));
        assert_eq!(sent.len(), 1, "{sent:?}");
        let settings = client.join_settings();
        assert_eq!(
            sent[0],
            mc_protocol::play::c2s::ClientSettings {
                locale: settings.locale.clone(),
                view_distance: 7,
                chat_visibility: settings.chat_visibility,
                chat_colors: settings.chat_colors,
                displayed_skin_parts: settings.displayed_skin_parts,
            }
        );
        // Back to 12: a change again, so one more C15.
        client.settings.render_distance_chunks = 12;
        client.handle_action(Action::SaveSettings);
        assert_eq!(c15s(&drained(&mut to_net)).len(), 1);
    }

    /// Item 16, the proxy half: a `JoinGame` after an in-game change (a server switch) resends
    /// the current settings, not the ones the client connected with, and always sends (it is
    /// vanilla's unconditional join-time send).
    #[test]
    fn a_later_join_game_resends_the_current_settings() {
        use mc_protocol::play::C2SPlay;
        let Some((mut client, _dir, mut to_net, to_game)) = client_in_game() else {
            return;
        };
        client.settings.render_distance_chunks = 5;
        client.handle_action(mc_ui::screens::Action::SaveSettings);
        let _ = drained(&mut to_net);
        to_game
            .try_send(crate::net::ToGame::Packet(
                mc_protocol::play::S2CPlay::JoinGame(mc_protocol::play::s2c::JoinGame {
                    entity_id: 2,
                    hardcore_mode: false,
                    game_mode: 0,
                    dimension: 0,
                    difficulty: 2,
                    max_players: 20,
                    level_type: "default".into(),
                    reduced_debug_info: false,
                }),
            ))
            .expect("channel");
        client.session.as_mut().expect("session").poll(16);
        let sent: Vec<_> = drained(&mut to_net)
            .into_iter()
            .filter_map(|packet| match packet {
                C2SPlay::ClientSettings(inner) => Some(inner),
                _ => None,
            })
            .collect();
        assert_eq!(sent.len(), 1, "the join-time send is unconditional");
        assert_eq!(sent[0].view_distance, 5);
        // And a save right after it, with nothing changed, sends nothing.
        client.handle_action(mc_ui::screens::Action::SaveSettings);
        assert!(drained(&mut to_net)
            .iter()
            .all(|packet| !matches!(packet, C2SPlay::ClientSettings(_))));
    }

    /// A stick whose NBT lists `names` under `key` (`CanDestroy` / `CanPlaceOn`).
    fn adventure_stick(key: &str, names: &[&str]) -> mc_protocol::ItemStack {
        let mut nbt = mc_nbt::NbtCompound::new();
        nbt.insert(
            key,
            mc_nbt::Nbt::list(
                names
                    .iter()
                    .map(|name| mc_nbt::Nbt::String((*name).to_owned()))
                    .collect(),
            ),
        );
        mc_protocol::ItemStack {
            item_id: mc_data::item_by_name("stick").expect("stick").0 as i16,
            count: 1,
            damage: 0,
            nbt: Some(nbt),
        }
    }

    fn block(name: &str) -> mc_data::BlockState {
        mc_data::block_by_name(name)
            .expect("a registered block")
            .default_state()
    }

    /// Item 18: `isDrawBlockOutline`'s three branches, plus Block Overlay's adventure option.
    #[test]
    fn adventure_and_spectator_outlines_follow_is_draw_block_outline() {
        use mc_protocol::play::s2c::GameMode;
        let Some((mut client, _dir, _to_net, _to_game)) = client_in_game() else {
            return;
        };
        let (stone, dirt, chest, sign) = (
            block("stone"),
            block("dirt"),
            block("chest"),
            block("standing_sign"),
        );
        let outline = |client: &Client, state, held: Option<&mc_protocol::ItemStack>, option| {
            Client::x4_draws_block_outline(
                client.session.as_ref().expect("session"),
                state,
                held,
                option,
            )
        };
        let set_mode = |client: &mut Client, mode: GameMode| {
            let local = &mut client.session.as_mut().expect("session").state_mut().local;
            local.game_mode = mode;
            local.abilities.configure_for_game_mode(mode);
        };

        // Survival: always (allowEdit).
        set_mode(&mut client, GameMode::Survival);
        assert!(outline(&client, stone, None, false));

        // Adventure: only what the held item's CanDestroy / CanPlaceOn names.
        set_mode(&mut client, GameMode::Adventure);
        assert!(
            !outline(&client, stone, None, false),
            "empty hand: no outline"
        );
        let breaks_stone = adventure_stick("CanDestroy", &["stone"]);
        assert!(outline(&client, stone, Some(&breaks_stone), false));
        assert!(!outline(&client, dirt, Some(&breaks_stone), false));
        let places_on_dirt = adventure_stick("CanPlaceOn", &["minecraft:dirt"]);
        assert!(outline(&client, dirt, Some(&places_on_dirt), false));
        assert!(!outline(&client, stone, Some(&places_on_dirt), false));
        // `Block.getBlockFromName` falls back to a numeric id.
        let numeric = adventure_stick("CanDestroy", &["1"]);
        assert!(outline(&client, stone, Some(&numeric), false));
        // Block Overlay's "Show in adventure mode": everything.
        assert!(outline(&client, dirt, None, true));

        // Spectator: only blocks with an IInventory tile entity, whatever the option says.
        set_mode(&mut client, GameMode::Spectator);
        assert!(outline(&client, chest, None, false));
        assert!(
            !outline(&client, sign, None, false),
            "a sign has a tile entity, no inventory"
        );
        assert!(!outline(&client, stone, None, true));

        // The option, read through the settings snapshot: off by default, needs the mod on.
        assert!(!client.x4_adventure_outline());
        client
            .mods
            .settings
            .set_bool(BLOCK_OVERLAY, ADVENTURE_OUTLINE, true);
        assert!(
            !client.x4_adventure_outline(),
            "Block Overlay itself is off"
        );
        client.mods.settings.set_enabled(BLOCK_OVERLAY, true);
        assert!(client.x4_adventure_outline());
    }

    /// Item 18 through the real `selection_box`: an adventure player aiming at stone with an
    /// empty hand gets no outline, then one with the option on.
    #[test]
    fn the_live_selection_box_honours_adventure_mode() {
        use mc_protocol::play::s2c::GameMode;
        let Some((mut client, _dir, _to_net, _to_game)) = client_in_game() else {
            return;
        };
        let pos = mc_math::BlockPos::new(3, 65, 4);
        {
            let session = client.session.as_mut().expect("session");
            let mut world = mc_render::terrain::fixture::blank_world(1);
            mc_render::terrain::fixture::place(&mut world, pos, "minecraft:stone");
            session.state_mut().world = world;
            let local = &mut session.state_mut().local;
            local.game_mode = GameMode::Adventure;
            local.abilities.configure_for_game_mode(GameMode::Adventure);
        }
        client.render_hit =
            crate::app::interact::HitResult::Block(crate::app::interact::BlockHit {
                pos,
                face: mc_math::Facing::Up,
                hit_vec: mc_math::DVec3::new(3.5, 66.0, 4.5),
            });
        assert!(
            client.selection_box().is_none(),
            "vanilla: no outline in adventure"
        );
        client.mods.settings.set_enabled(BLOCK_OVERLAY, true);
        client
            .mods
            .settings
            .set_bool(BLOCK_OVERLAY, ADVENTURE_OUTLINE, true);
        assert!(client.selection_box().is_some(), "the option outlines it");
        // Control: survival outlines it with the option off.
        client
            .mods
            .settings
            .set_bool(BLOCK_OVERLAY, ADVENTURE_OUTLINE, false);
        let local = &mut client.session.as_mut().expect("session").state_mut().local;
        local.game_mode = GameMode::Survival;
        local.abilities.configure_for_game_mode(GameMode::Survival);
        assert!(client.selection_box().is_some());
    }

    /// Give the session loaded (blank) chunks around the origin: like vanilla's spawn queue, the
    /// tracker parks an entity spawned into an unloaded chunk, and the pick never sees it.
    fn load_blank_world(client: &mut Client) {
        client.session.as_mut().expect("session").state_mut().world =
            mc_render::terrain::fixture::blank_world(1);
    }

    /// Deliver server packets to the test session and run its packet pass.
    fn deliver(
        client: &mut Client,
        to_game: &tokio::sync::mpsc::Sender<crate::net::ToGame>,
        packets: Vec<mc_protocol::play::S2CPlay>,
    ) {
        for packet in packets {
            to_game
                .try_send(crate::net::ToGame::Packet(packet))
                .expect("channel");
        }
        client.session.as_mut().expect("session").poll(64);
    }

    /// One attack press and one use press, each in its own tick, with an empty hand; returns
    /// the packets each produced.
    fn punch_then_use(
        client: &mut Client,
        to_net: &mut ToNet,
    ) -> (
        Vec<mc_protocol::play::C2SPlay>,
        Vec<mc_protocol::play::C2SPlay>,
    ) {
        // `handlePlayerPosLook` turns `rotationYaw` but not `rotationYawHead`, the yaw a click
        // aims with (L7); the ticks before a real click copy it. Model
        // them, so the S08 that aimed the player here also aims the click.
        {
            let session = client.session.as_mut().expect("session");
            let (yaw, pitch) = session
                .predictor()
                .map(|predictor| (predictor.state.yaw, predictor.state.pitch))
                .expect("JoinGame built the predictor");
            session.face_local_player(yaw, pitch);
        }
        let raw = InputState::new();
        let mut keys = client.settings.keys.clone();
        client.handle_game_input(&raw, &mut keys);
        let _ = drained(to_net);
        client.update_hit_result();
        keys.set_state(Input::MOUSE_LEFT, true);
        keys.set_state(Input::MOUSE_LEFT, false);
        client.handle_game_input(&raw, &mut keys);
        let attack = drained(to_net);
        client.update_hit_result();
        keys.set_state(Input::MOUSE_RIGHT, true);
        keys.set_state(Input::MOUSE_RIGHT, false);
        client.handle_game_input(&raw, &mut keys);
        let used = drained(to_net);
        (attack, used)
    }

    /// The C02s in `sent`, as `(entity id, action)`.
    fn use_entity(
        sent: &[mc_protocol::play::C2SPlay],
    ) -> Vec<(i32, mc_protocol::play::c2s::UseEntityAction)> {
        sent.iter()
            .filter_map(|packet| match packet {
                mc_protocol::play::C2SPlay::UseEntity(inner) => {
                    Some((inner.entity_id, inner.action))
                }
                _ => None,
            })
            .collect()
    }

    /// Item 3 ("can't punch NPCs" on a server): the NPC shapes servers build, end to end from
    /// their spawn packets to the C02s mc-rs puts on the wire, against what 1.8.9's
    /// `getMouseOver` / `PlayerControllerMP` would send for the same packets:
    ///
    /// * a player NPC (`S0C`) whose tab entry was added and then removed (as NPC plugins do);
    /// * an invisible, NoAI villager (vanilla picks invisible entities);
    /// * a zombie under a marker-armour-stand hologram (markers are not pickable);
    /// * a non-marker invisible armour stand in front of an NPC, which vanilla *does* pick,
    ///   and whose right-click sends only `INTERACT_AT` (`EntityArmorStand.interactAt` returns
    ///   true on the client).
    ///
    /// The attack is always `C0A` then `C02 ATTACK`, sent whatever the client thinks of the
    /// target's health, and the use is `C02 INTERACT_AT` then `C02 INTERACT`.
    #[test]
    fn npc_shapes_are_picked_and_clicked_exactly_as_vanilla_would() {
        use mc_protocol::play::c2s::UseEntityAction;
        use mc_protocol::play::{s2c, C2SPlay, S2CPlay};
        use mc_protocol::{Angle, MetadataEntry, MetadataValue};
        let Some((mut client, _dir, mut to_net, to_game)) = client_in_game() else {
            return;
        };
        load_blank_world(&mut client);
        let fixed = |v: f64| (v * 32.0).floor() as i32;
        // Stand at (0.5, 64, 0.5) looking due south (+Z), level: the eye is at y 65.62.
        let look_south = || {
            S2CPlay::PlayerPositionLook(s2c::PlayerPositionLook {
                x: 0.5,
                y: 64.0,
                z: 0.5,
                yaw: 0.0,
                pitch: 0.0,
                flags: 0,
            })
        };
        let attack_and_interact = |id: i32| {
            (
                vec![(id, UseEntityAction::Attack)],
                vec![
                    (
                        id,
                        UseEntityAction::InteractAt {
                            target_x: 0.0,
                            target_y: 0.0,
                            target_z: 0.0,
                        },
                    ),
                    (id, UseEntityAction::Interact),
                ],
            )
        };
        // INTERACT_AT's hit vector depends on the exact intercept; compare actions only.
        let actions = |c02s: Vec<(i32, UseEntityAction)>| {
            c02s.into_iter()
                .map(|(id, action)| match action {
                    UseEntityAction::InteractAt { .. } => (
                        id,
                        UseEntityAction::InteractAt {
                            target_x: 0.0,
                            target_y: 0.0,
                            target_z: 0.0,
                        },
                    ),
                    other => (id, other),
                })
                .collect::<Vec<_>>()
        };

        // 1. A player NPC 2 blocks ahead whose tab entry is added, used, then removed.
        let npc = uuid::Uuid::from_u128(0x0123_4567_89ab_cdef_0123_4567_89ab_cdef);
        deliver(
            &mut client,
            &to_game,
            vec![
                look_south(),
                S2CPlay::PlayerListItem(s2c::PlayerListItem::AddPlayer(vec![
                    s2c::AddPlayerEntry {
                        uuid: npc,
                        username: "Play SkyWars".into(),
                        properties: vec![],
                        game_mode: s2c::GameMode::Survival,
                        ping: 0,
                        display_name: None,
                    },
                ])),
                S2CPlay::SpawnPlayer(s2c::SpawnPlayer {
                    entity_id: 500,
                    player_uuid: npc,
                    x: fixed(0.5),
                    y: fixed(64.0),
                    z: fixed(2.5),
                    yaw: Angle::from_degrees(180.0),
                    pitch: Angle(0),
                    current_item: 0,
                    metadata: Vec::new(),
                }),
                S2CPlay::PlayerListItem(s2c::PlayerListItem::RemovePlayer(vec![npc])),
            ],
        );
        let (attack, used) = punch_then_use(&mut client, &mut to_net);
        assert!(
            matches!(
                attack.as_slice(),
                [C2SPlay::Animation(_), C2SPlay::UseEntity(_)]
            ),
            "player NPC: swing then attack, {attack:?}"
        );
        let (want_attack, want_use) = attack_and_interact(500);
        assert_eq!(use_entity(&attack), want_attack);
        assert_eq!(actions(use_entity(&used)), want_use, "{used:?}");
        deliver(
            &mut client,
            &to_game,
            vec![S2CPlay::DestroyEntities(s2c::DestroyEntities {
                entity_ids: vec![500],
            })],
        );

        // 2. An invisible NoAI villager (watched 0 = 0x20 invisible, 15 = NoAI).
        deliver(
            &mut client,
            &to_game,
            vec![S2CPlay::SpawnMob(s2c::SpawnMob {
                entity_id: 501,
                entity_type: 120,
                x: fixed(0.5),
                y: fixed(64.0),
                z: fixed(2.5),
                yaw: Angle(0),
                pitch: Angle(0),
                head_pitch: Angle(0),
                velocity_x: 0,
                velocity_y: 0,
                velocity_z: 0,
                metadata: vec![
                    MetadataEntry {
                        index: 0,
                        value: MetadataValue::Byte(0x20),
                    },
                    MetadataEntry {
                        index: 15,
                        value: MetadataValue::Byte(1),
                    },
                ],
            })],
        );
        let (attack, used) = punch_then_use(&mut client, &mut to_net);
        let (want_attack, want_use) = attack_and_interact(501);
        assert_eq!(
            use_entity(&attack),
            want_attack,
            "invisible villager: {attack:?}"
        );
        assert_eq!(actions(use_entity(&used)), want_use, "{used:?}");
        deliver(
            &mut client,
            &to_game,
            vec![S2CPlay::DestroyEntities(s2c::DestroyEntities {
                entity_ids: vec![501],
            })],
        );

        // 3. A zombie with a marker hologram stand at eye height right in front of its face.
        let stand = |id: i32, z: f64, flags: i8| {
            vec![
                S2CPlay::SpawnObject(s2c::SpawnObject {
                    entity_id: id,
                    object_type: 78,
                    x: fixed(0.5),
                    y: fixed(65.0),
                    z: fixed(z),
                    pitch: Angle(0),
                    yaw: Angle(0),
                    data: 0,
                    velocity: None,
                }),
                S2CPlay::EntityMetadata(s2c::EntityMetadata {
                    entity_id: id,
                    metadata: vec![
                        MetadataEntry {
                            index: 0,
                            value: MetadataValue::Byte(0x20),
                        },
                        MetadataEntry {
                            index: 10,
                            value: MetadataValue::Byte(flags),
                        },
                    ],
                }),
            ]
        };
        let mut packets = vec![S2CPlay::SpawnMob(s2c::SpawnMob {
            entity_id: 502,
            entity_type: 54,
            x: fixed(0.5),
            y: fixed(64.0),
            z: fixed(2.5),
            yaw: Angle(0),
            pitch: Angle(0),
            head_pitch: Angle(0),
            velocity_x: 0,
            velocity_y: 0,
            velocity_z: 0,
            metadata: Vec::new(),
        })];
        packets.extend(stand(503, 1.9, 0x10));
        deliver(&mut client, &to_game, packets);
        let (attack, used) = punch_then_use(&mut client, &mut to_net);
        let (want_attack, want_use) = attack_and_interact(502);
        assert_eq!(
            use_entity(&attack),
            want_attack,
            "marker hologram: {attack:?}"
        );
        assert_eq!(actions(use_entity(&used)), want_use, "{used:?}");

        // 4. The same stand without the marker bit is a real, pickable entity in vanilla too:
        // it takes the hit, and its right-click is INTERACT_AT alone.
        deliver(
            &mut client,
            &to_game,
            vec![S2CPlay::EntityMetadata(s2c::EntityMetadata {
                entity_id: 503,
                metadata: vec![MetadataEntry {
                    index: 10,
                    value: MetadataValue::Byte(0),
                }],
            })],
        );
        let (attack, used) = punch_then_use(&mut client, &mut to_net);
        assert_eq!(use_entity(&attack), vec![(503, UseEntityAction::Attack)]);
        assert_eq!(
            actions(use_entity(&used)),
            vec![(
                503,
                UseEntityAction::InteractAt {
                    target_x: 0.0,
                    target_y: 0.0,
                    target_z: 0.0
                }
            )],
            "a non-marker stand consumes interactAt: {used:?}"
        );
    }

    /// Item 3: the pick's cheap candidate reject is exact on the entity's grown box. The old
    /// test (feet position within reach + 4) dropped the near parts of big entities.
    #[test]
    fn the_candidate_reject_keeps_every_box_the_entity_ray_can_reach() {
        use crate::app::interact::box_within_reach;
        use mc_math::{Aabb, DVec3};
        let eye = DVec3::new(0.5, 65.62, 0.5);
        // A giant (3.6 x 10.8) whose feet are 11.9 blocks away but whose top is 0.94 away.
        let giant = Aabb::new(DVec3::new(-1.3, 54.0, 1.2), DVec3::new(2.3, 64.8, 4.8));
        assert!(
            DVec3::new(0.5, 54.0, 3.0).distance(eye) > 4.5 + 4.0,
            "the old reject"
        );
        assert!(box_within_reach(giant, 0.1, eye, 4.5));
        // Just beyond the ray, and just within it thanks to the 0.1 border.
        let far = Aabb::new(DVec3::new(0.2, 65.0, 5.2), DVec3::new(0.8, 66.8, 5.8));
        assert!(!box_within_reach(far, 0.0, eye, 4.5));
        let edge = Aabb::new(DVec3::new(0.2, 65.0, 5.05), DVec3::new(0.8, 66.8, 5.6));
        assert!(!box_within_reach(edge, 0.0, eye, 4.5));
        assert!(box_within_reach(edge, 0.1, eye, 4.5));
        // The eye inside a box is always within reach.
        let around = Aabb::new(DVec3::new(0.0, 64.0, 0.0), DVec3::new(1.0, 66.0, 1.0));
        assert!(box_within_reach(around, 0.0, eye, 0.0));
    }

    /// Item 3, end to end: a giant's head right in front of a player is hit (vanilla's box query
    /// finds it), where the old feet-distance reject skipped it.
    #[test]
    fn a_giant_npc_is_pickable_by_the_part_in_reach() {
        use mc_protocol::play::c2s::UseEntityAction;
        use mc_protocol::play::{s2c, S2CPlay};
        use mc_protocol::Angle;
        let Some((mut client, _dir, mut to_net, to_game)) = client_in_game() else {
            return;
        };
        load_blank_world(&mut client);
        deliver(
            &mut client,
            &to_game,
            vec![
                S2CPlay::PlayerPositionLook(s2c::PlayerPositionLook {
                    x: 0.5,
                    y: 64.0,
                    z: 0.5,
                    yaw: 0.0,
                    pitch: 30.0,
                    flags: 0,
                }),
                S2CPlay::SpawnMob(s2c::SpawnMob {
                    entity_id: 600,
                    entity_type: 53,
                    x: 16,
                    y: 54 * 32,
                    z: 96,
                    yaw: Angle(0),
                    pitch: Angle(0),
                    head_pitch: Angle(0),
                    velocity_x: 0,
                    velocity_y: 0,
                    velocity_z: 0,
                    metadata: Vec::new(),
                }),
            ],
        );
        let (attack, _) = punch_then_use(&mut client, &mut to_net);
        assert_eq!(
            use_entity(&attack),
            vec![(600, UseEntityAction::Attack)],
            "{attack:?}"
        );
    }

    /// A screen that opens and closes inside one tick still re-applies held keys: no tick ever
    /// saw it open, only its pending unpress.
    #[test]
    fn a_screen_opened_and_closed_between_two_ticks_still_represses() {
        let Some((mut client, _dir, _to_net, _to_game)) = client_in_game() else {
            return;
        };
        let mut keys = client.settings.keys.clone();
        let mut raw = InputState::new();
        held(&mut raw, &mut keys, KeyCode::KeyD);
        client.handle_game_input(&raw, &mut keys);
        client.open_chat("");
        client.show_screen(None);
        client.handle_game_input(&raw, &mut keys);
        assert!(keys.is_down(keys::RIGHT));
        // A key released while the screen was open is not re-pressed.
        raw.on_key(KeyCode::KeyD, false, false);
        client.open_chat("");
        client.show_screen(None);
        client.handle_game_input(&raw, &mut keys);
        assert!(!keys.is_down(keys::RIGHT));
    }

    // ---- Eating stops mining ---------------------------
    // ---- Now 1.7 Animations' "Using items stops mining",
    // ---- for sword blocking and bow drawing too ---------------------------------------------

    use crate::app::interact::{right_click_starts_use, UseAim, UseGuards};
    use mc_protocol::play::c2s::PlayerDiggingStatus;
    use mc_protocol::play::C2SPlay;

    /// Dirt under the player's feet, the block every B2 sequence mines.
    const DIRT: mc_math::BlockPos = mc_math::BlockPos::new(0, 63, 0);

    /// 1.7 Animations on (it starts off) with "Using items stops mining" at `option`, or at its
    /// default (on) for `None`.
    fn use_stops_mining(client: &mut Client, option: Option<bool>) {
        client.mods.settings.set_enabled(OLD_ANIMATIONS, true);
        if let Some(on) = option {
            client
                .mods
                .settings
                .set_bool(OLD_ANIMATIONS, USE_STOPS_MINING, on);
        }
    }

    fn stack(item_id: i16, damage: i16) -> mc_protocol::ItemStack {
        mc_protocol::ItemStack {
            item_id,
            count: 4,
            damage,
            nbt: None,
        }
    }

    /// A survival client standing on [`DIRT`] at (0.5, 64, 0.5) and looking straight down at it,
    /// with `held` in hotbar slot 0 (selected) and `food` hunger.
    fn mining_client(
        held: mc_protocol::ItemStack,
        food: i32,
    ) -> Option<(
        Client,
        tempfile::TempDir,
        ToNet,
        tokio::sync::mpsc::Sender<crate::net::ToGame>,
    )> {
        mining_client_on(held, food, block("dirt"))
    }

    /// [`mining_client`] with `target` in place of the dirt.
    fn mining_client_on(
        held: mc_protocol::ItemStack,
        food: i32,
        target: mc_data::BlockState,
    ) -> Option<(
        Client,
        tempfile::TempDir,
        ToNet,
        tokio::sync::mpsc::Sender<crate::net::ToGame>,
    )> {
        use mc_protocol::play::{s2c, S2CPlay};
        let (mut client, dir, mut to_net, to_game) = client_in_game()?;
        load_blank_world(&mut client);
        client
            .session
            .as_mut()
            .expect("session")
            .state_mut()
            .world
            .set_block(DIRT, target);
        deliver(
            &mut client,
            &to_game,
            vec![S2CPlay::PlayerPositionLook(s2c::PlayerPositionLook {
                x: 0.5,
                y: 64.0,
                z: 0.5,
                yaw: 0.0,
                pitch: 90.0,
                flags: 0,
            })],
        );
        client
            .session
            .as_mut()
            .expect("session")
            .face_local_player(0.0, 90.0);
        client.player_hud.selected_slot = 0;
        client
            .player_container
            .set_slot_contents(super::super::super::HOTBAR_SLOT_OFFSET, Some(held));
        client.refresh_hotbar();
        client.player_hud.food = food;
        let _ = drained(&mut to_net);
        Some((client, dir, to_net, to_game))
    }

    /// One `runTick` input pass: the click timers, the pick, `handle_game_input`, the crack
    /// mirror. Returns what went on the wire.
    fn tick(client: &mut Client, keys: &mut KeyBindings, to_net: &mut ToNet) -> Vec<C2SPlay> {
        client.controller.tick();
        client.update_hit_result();
        client.handle_game_input(&InputState::new(), keys);
        client.mirror_local_break_progress();
        drained(to_net)
    }

    fn digging(status: PlayerDiggingStatus, packets: &[C2SPlay]) -> Vec<mc_math::BlockPos> {
        packets
            .iter()
            .filter_map(|packet| match packet {
                C2SPlay::PlayerDigging(dig) if dig.status == status => Some(dig.position),
                _ => None,
            })
            .collect()
    }

    /// The local player's crack stage on [`DIRT`], if any.
    fn crack(client: &Client) -> Option<i32> {
        client
            .damaged_blocks
            .get(&1)
            .filter(|damaged| damaged.pos == DIRT)
            .map(|damaged| i32::from(damaged.stage))
    }

    /// Mine [`DIRT`] for two ticks with attack held, then press use (attack still held), with
    /// 1.7 Animations on and "Using items stops mining" at `option` ([`use_stops_mining`]).
    /// Returns the client mid-dig and the use tick's packets.
    fn mine_then_use(
        held: mc_protocol::ItemStack,
        food: i32,
        option: Option<bool>,
    ) -> Option<(Client, KeyBindings, ToNet, Vec<C2SPlay>, tempfile::TempDir)> {
        mine_then_use_on(held, food, option, block("dirt"), |_| {})
    }

    /// [`mine_then_use`] on `target`, with `setup` run just before the use press (arrows in the
    /// inventory, sneaking...).
    fn mine_then_use_on(
        held: mc_protocol::ItemStack,
        food: i32,
        option: Option<bool>,
        target: mc_data::BlockState,
        setup: impl FnOnce(&mut Client),
    ) -> Option<(Client, KeyBindings, ToNet, Vec<C2SPlay>, tempfile::TempDir)> {
        let (mut client, dir, mut to_net, _to_game) = mining_client_on(held, food, target)?;
        use_stops_mining(&mut client, option);
        let mut keys = client.settings.keys.clone();
        keys.set_state(Input::MOUSE_LEFT, true);
        let first = tick(&mut client, &mut keys, &mut to_net);
        assert_eq!(
            digging(PlayerDiggingStatus::StartDestroyBlock, &first),
            [DIRT],
            "{first:?}"
        );
        // Dig on until the crack shows (its first stage needs a tenth of the block's damage) and
        // the dig is a held one (`USE_STOPS_MINING_MIN_DIG_TICKS`).
        for _ in 0..40 {
            let _ = tick(&mut client, &mut keys, &mut to_net);
            if crack(&client).is_some()
                && client.controller.dig_ticks() >= USE_STOPS_MINING_MIN_DIG_TICKS
            {
                break;
            }
        }
        assert_eq!(client.controller.hitting_block(), Some(DIRT));
        assert!(crack(&client).is_some(), "the dig shows its crack");
        setup(&mut client);
        let _ = drained(&mut to_net);
        keys.set_state(Input::MOUSE_RIGHT, true);
        let used = tick(&mut client, &mut keys, &mut to_net);
        Some((client, keys, to_net, used, dir))
    }

    /// The untargeted `C08` (`sendUseItem`) and the targeted one (`onPlayerRightClick`).
    fn placements(packets: &[C2SPlay]) -> (usize, usize) {
        let mut untargeted = 0;
        let mut targeted = 0;
        for packet in packets {
            if let C2SPlay::PlayerBlockPlacement(place) = packet {
                if place.position == mc_math::BlockPos::new(-1, -1, -1) {
                    assert_eq!(place.placed_block_direction, 255);
                    untargeted += 1;
                } else {
                    assert_eq!(place.position, DIRT);
                    targeted += 1;
                }
            }
        }
        (untargeted, targeted)
    }

    /// The predicate: eat or drink actions whose own `onItemRightClick` guard would really start
    /// a use. Hunger, golden apples, `disableDamage` and splash potions decide it.
    #[test]
    fn only_a_use_that_really_starts_counts() {
        let sky = UseAim::Nothing;
        let hungry = UseGuards {
            food_level: 15,
            ..UseGuards::default()
        };
        let full = UseGuards {
            food_level: 20,
            ..UseGuards::default()
        };
        let creative = UseGuards {
            food_level: 15,
            disable_damage: true,
            creative: true,
            ..UseGuards::default()
        };
        let armed = UseGuards {
            has_arrow: true,
            ..hungry
        };
        let steak = stack(364, 0);
        let golden_apple = stack(322, 0);
        let stew = stack(282, 0);
        let milk = stack(335, 0);
        let drink = stack(373, 8_193);
        let splash = stack(373, 16_385);
        let sword = stack(276, 0);
        let bow = stack(261, 0);
        // Eating and drinking: B2's table, unchanged.
        assert!(right_click_starts_use(&steak, hungry, sky));
        assert!(right_click_starts_use(&stew, hungry, sky));
        assert!(
            !right_click_starts_use(&steak, full, sky),
            "canEat: not hungry"
        );
        assert!(
            right_click_starts_use(&golden_apple, full, sky),
            "golden apples are always edible"
        );
        assert!(
            !right_click_starts_use(&golden_apple, creative, sky),
            "disableDamage"
        );
        assert!(right_click_starts_use(&milk, full, sky));
        assert!(right_click_starts_use(&drink, full, sky));
        assert!(
            !right_click_starts_use(&splash, hungry, sky),
            "thrown, not drunk"
        );
        // Swords always block; a bow draws only with an arrow, or in creative.
        assert!(right_click_starts_use(&sword, full, sky));
        assert!(right_click_starts_use(&sword, creative, sky));
        assert!(!right_click_starts_use(&bow, hungry, sky), "no arrow");
        assert!(right_click_starts_use(&bow, armed, sky));
        assert!(right_click_starts_use(&bow, creative, sky));
        for other in [stack(368, 0), stack(3, 0), stack(346, 0), stack(332, 0)] {
            assert!(
                !right_click_starts_use(&other, armed, sky),
                "pearl, dirt, rod, snowball: {other:?}"
            );
        }
        // The aim: a block that takes the click, or an entity, never starts a use.
        for held in [&sword, &steak, &drink] {
            assert!(right_click_starts_use(
                held,
                hungry,
                UseAim::Block { consumed: false }
            ));
            assert!(!right_click_starts_use(
                held,
                hungry,
                UseAim::Block { consumed: true }
            ));
            assert!(!right_click_starts_use(held, hungry, UseAim::Entity));
        }
    }

    /// B2 end to end through `handle_game_input`, against vanilla. With the option on (its
    /// default once 1.7 Animations is on), pressing use with steak while mining sends, in that
    /// one tick, exactly the
    /// packets vanilla sends across the two ticks of "release attack, then right-click":
    /// `ABORT_DESTROY_BLOCK` at the dug block, then the targeted and the untargeted C08, and
    /// nothing else (no C0A swing, no START or STOP). The crack is cleared, eating starts, and
    /// while it lasts holding attack sends nothing. Letting go of use ends it with
    /// `RELEASE_USE_ITEM`, and the still-held attack starts the dig again with a fresh START.
    ///
    /// The negative control is the same input with the option off: the use press is dropped and
    /// the dig goes on (vanilla's `getIsHittingBlock` gate); the ABORT only comes when attack is
    /// released, and the C08s a tick after that.
    #[test]
    fn eating_stops_mining_in_one_tick_with_vanillas_own_packets() {
        // ---- the option on (the default) ----
        let Some((mut client, mut keys, mut to_net, used, _dir)) =
            mine_then_use(stack(364, 0), 15, None)
        else {
            return;
        };
        assert!(
            client.mods.settings.bool(OLD_ANIMATIONS, USE_STOPS_MINING),
            "on by default in 1.7 Animations"
        );
        assert_eq!(
            digging(PlayerDiggingStatus::AbortDestroyBlock, &used),
            [DIRT],
            "{used:?}"
        );
        assert!(
            matches!(&used[0], C2SPlay::PlayerDigging(dig)
                if dig.status == PlayerDiggingStatus::AbortDestroyBlock
                    && dig.face == mc_math::Facing::Down),
            "the ABORT comes first, facing down as resetBlockRemoving sends it: {used:?}"
        );
        assert_eq!(placements(&used), (1, 1), "{used:?}");
        assert_eq!(used.len(), 3, "nothing else: {used:?}");
        assert_eq!(client.controller.hitting_block(), None);
        assert!(
            client.controller.is_using_item(),
            "eating started this tick"
        );
        assert_eq!(crack(&client), None, "the crack is gone");
        let one_tick = used;

        // Eating: attack and use held, nothing at all goes out (vanilla's isUsingItem gate).
        for _ in 0..5 {
            let quiet = tick(&mut client, &mut keys, &mut to_net);
            assert!(quiet.is_empty(), "{quiet:?}");
        }
        assert!(client.controller.is_using_item());

        // Use released, attack held: RELEASE_USE_ITEM, then the dig carries on from zero damage
        // the way vanilla's own controller does after any reset on the same block:
        // `isHittingPosition` ignores `isHittingBlock`, so `onPlayerDamageBlock` resumes without
        // a new START (as after glancing off a block and back), swinging each tick.
        keys.set_state(Input::MOUSE_RIGHT, false);
        let resumed = tick(&mut client, &mut keys, &mut to_net);
        assert!(
            matches!(&resumed[0], C2SPlay::PlayerDigging(dig)
                if dig.status == PlayerDiggingStatus::ReleaseUseItem),
            "{resumed:?}"
        );
        assert!(
            matches!(resumed.last(), Some(C2SPlay::Animation(_))),
            "{resumed:?}"
        );
        assert!(digging(PlayerDiggingStatus::AbortDestroyBlock, &resumed).is_empty());
        assert!(digging(PlayerDiggingStatus::StartDestroyBlock, &resumed).is_empty());
        assert!(client.controller.damage() > 0.0, "the dig resumed");

        // ---- the negative control: the option off is vanilla ----
        let Some((mut client, mut keys, mut to_net, used, _dir)) =
            mine_then_use(stack(364, 0), 15, Some(false))
        else {
            return;
        };
        assert!(digging(PlayerDiggingStatus::AbortDestroyBlock, &used).is_empty());
        assert_eq!(placements(&used), (0, 0), "the press is dropped: {used:?}");
        assert_eq!(
            client.controller.hitting_block(),
            Some(DIRT),
            "still mining"
        );
        assert!(!client.controller.is_using_item());
        // Vanilla: release attack (ABORT), and the held use eats on the next tick.
        keys.set_state(Input::MOUSE_LEFT, false);
        let released = tick(&mut client, &mut keys, &mut to_net);
        assert_eq!(
            digging(PlayerDiggingStatus::AbortDestroyBlock, &released),
            [DIRT]
        );
        assert_eq!(placements(&released), (0, 0), "{released:?}");
        let next = tick(&mut client, &mut keys, &mut to_net);
        assert!(client.controller.is_using_item());
        let two_ticks: Vec<C2SPlay> = released.into_iter().chain(next).collect();
        assert_eq!(
            one_tick, two_ticks,
            "the same packets as vanilla's release-then-right-click, in the same order"
        );
    }

    /// With 1.7 Animations off (its default) the option does nothing, whatever it says, and
    /// Modern Keybinds no longer has any say: vanilla drops the press. A sword and steak, with
    /// Modern Keybinds on and off.
    #[test]
    fn animations_off_is_vanilla() {
        for (held, modern) in [
            (stack(364, 0), true),
            (stack(276, 0), true),
            (stack(276, 0), false),
        ] {
            let Some((mut client, dir, mut to_net, _to_game)) = mining_client(held.clone(), 15)
            else {
                return;
            };
            assert!(
                !client.mods.settings.enabled(OLD_ANIMATIONS),
                "off by default"
            );
            assert!(client.mods.settings.bool(OLD_ANIMATIONS, USE_STOPS_MINING));
            client.mods.settings.set_enabled(MODERN_KEYBINDS, modern);
            let mut keys = client.settings.keys.clone();
            keys.set_state(Input::MOUSE_LEFT, true);
            let _ = tick(&mut client, &mut keys, &mut to_net);
            keys.set_state(Input::MOUSE_RIGHT, true);
            let used = tick(&mut client, &mut keys, &mut to_net);
            assert_eq!(placements(&used), (0, 0), "{held:?}: {used:?}");
            assert!(digging(PlayerDiggingStatus::AbortDestroyBlock, &used).is_empty());
            assert_eq!(client.controller.hitting_block(), Some(DIRT));
            drop(dir);
        }
    }

    /// A press that would not start a use leaves the dig alone, exactly as vanilla: steak at
    /// full hunger, a splash potion, a bow without arrows. A golden apple at full hunger, a
    /// drinkable potion, milk, a sword do stop it.
    #[test]
    fn only_a_press_that_starts_a_use_stops_the_dig() {
        for (held, food, stops) in [
            (stack(364, 0), 20, false),
            (stack(373, 16_385), 15, false),
            (stack(261, 0), 15, false),
            (stack(276, 0), 15, true),
            (stack(322, 0), 20, true),
            (stack(373, 8_193), 20, true),
            (stack(335, 0), 20, true),
        ] {
            let Some((client, _keys, _to_net, used, _dir)) =
                mine_then_use(held.clone(), food, None)
            else {
                return;
            };
            let aborted = !digging(PlayerDiggingStatus::AbortDestroyBlock, &used).is_empty();
            assert_eq!(aborted, stops, "{held:?} at food {food}: {used:?}");
            assert_eq!(client.controller.hitting_block().is_none(), stops);
            if stops {
                assert!(client.controller.is_using_item(), "{held:?}");
                assert_eq!(placements(&used), (1, 1), "{held:?}: {used:?}");
            } else {
                assert!(!client.controller.is_using_item(), "{held:?}");
            }
        }
    }

    /// When the server finishes the meal (status 9) with both buttons still held, the next tick
    /// digs again: vanilla's controller resumes the same block (no new START, a swing, damage
    /// from zero), exactly as it does after any other reset with attack held on that block.
    #[test]
    fn a_finished_meal_with_attack_held_restarts_the_dig() {
        let Some((mut client, mut keys, mut to_net, used, _dir)) =
            mine_then_use(stack(364, 0), 19, None)
        else {
            return;
        };
        assert_eq!(placements(&used), (1, 1));
        // Status 9 for the local player (entity 1): `EntityPlayer.onItemUseFinish`.
        client.react(&crate::sync::ClientEvent::EntityStatus {
            network_id: 1,
            opcode: 9,
        });
        assert!(!client.controller.is_using_item(), "the meal is over");
        assert_eq!(client.controller.damage(), 0.0);
        let next = tick(&mut client, &mut keys, &mut to_net);
        assert_eq!(next.len(), 1, "{next:?}");
        assert!(matches!(next[0], C2SPlay::Animation(_)), "{next:?}");
        assert!(client.controller.damage() > 0.0, "the dig resumed");
    }


    const SWORD: i16 = 276;
    const BOW: i16 = 261;
    const ARROW: i16 = 262;

    /// An arrow stack in the first main-inventory slot (window 0 slot 9), where
    /// `InventoryPlayer.hasItem(Items.arrow)` finds it.
    fn give_arrows(client: &mut Client) {
        client
            .player_container
            .set_slot_contents(9, Some(stack(ARROW, 0)));
    }

    /// Sword blocking and bow drawing, end to end against vanilla, as B2's eating test: in one
    /// tick exactly ABORT (facing down), the targeted C08 and the untargeted C08, nothing else;
    /// the crack is gone and the use has started. While it lasts, attack held sends nothing.
    /// Letting go of use sends RELEASE_USE_ITEM (the bow fires) and the still-held attack
    /// resumes the dig on the same block without a START, as vanilla's controller does after
    /// any reset there.
    ///
    /// Negative control, the same input with the option off: the press is dropped and the dig
    /// goes on; vanilla's own "let go of attack, keep use held" then sends the same packets in
    /// the same order, over two ticks.
    #[test]
    fn blocking_and_drawing_stop_mining_in_one_tick_with_vanillas_own_packets() {
        for item in [SWORD, BOW] {
            let arm = |client: &mut Client| {
                if item == BOW {
                    give_arrows(client);
                }
            };
            let Some((mut client, mut keys, mut to_net, used, _dir)) =
                mine_then_use_on(stack(item, 0), 20, None, block("dirt"), arm)
            else {
                return;
            };
            assert!(
                matches!(&used[0], C2SPlay::PlayerDigging(dig)
                    if dig.status == PlayerDiggingStatus::AbortDestroyBlock
                        && dig.position == DIRT
                        && dig.face == mc_math::Facing::Down),
                "{item}: the ABORT comes first: {used:?}"
            );
            assert_eq!(placements(&used), (1, 1), "{item}: {used:?}");
            assert_eq!(used.len(), 3, "{item}: nothing else: {used:?}");
            assert!(
                matches!(&used[1], C2SPlay::PlayerBlockPlacement(place) if place.position == DIRT),
                "{item}: then the targeted C08: {used:?}"
            );
            assert_eq!(client.controller.hitting_block(), None);
            assert!(client.controller.is_using_item(), "{item}: the use started");
            assert_eq!(
                client.controller.item_in_use().map(|stack| stack.item_id),
                Some(item)
            );
            assert_eq!(crack(&client), None, "{item}: the crack is gone");
            let one_tick = used;
            for _ in 0..5 {
                let quiet = tick(&mut client, &mut keys, &mut to_net);
                assert!(quiet.is_empty(), "{item}: {quiet:?}");
            }
            keys.set_state(Input::MOUSE_RIGHT, false);
            let resumed = tick(&mut client, &mut keys, &mut to_net);
            assert!(
                matches!(&resumed[0], C2SPlay::PlayerDigging(dig)
                    if dig.status == PlayerDiggingStatus::ReleaseUseItem),
                "{item}: {resumed:?}"
            );
            assert!(digging(PlayerDiggingStatus::StartDestroyBlock, &resumed).is_empty());
            assert!(digging(PlayerDiggingStatus::AbortDestroyBlock, &resumed).is_empty());
            assert!(client.controller.damage() > 0.0, "{item}: the dig resumed");

            // ---- negative control: the option off is vanilla ----
            let Some((mut client, mut keys, mut to_net, used, _dir)) =
                mine_then_use_on(stack(item, 0), 20, Some(false), block("dirt"), arm)
            else {
                return;
            };
            assert!(
                used.is_empty() || placements(&used) == (0, 0),
                "{item}: {used:?}"
            );
            assert!(digging(PlayerDiggingStatus::AbortDestroyBlock, &used).is_empty());
            assert_eq!(
                client.controller.hitting_block(),
                Some(DIRT),
                "still mining"
            );
            assert!(!client.controller.is_using_item());
            keys.set_state(Input::MOUSE_LEFT, false);
            let released = tick(&mut client, &mut keys, &mut to_net);
            let next = tick(&mut client, &mut keys, &mut to_net);
            assert!(client.controller.is_using_item(), "{item}");
            let two_ticks: Vec<C2SPlay> = released.into_iter().chain(next).collect();
            assert_eq!(
                one_tick, two_ticks,
                "{item}: vanilla's release-then-right-click, packet for packet"
            );
        }
    }

    /// A bow with no arrow cannot draw, so it leaves the dig alone; give it an arrow and the
    /// same press stops the dig.
    #[test]
    fn a_bow_stops_mining_only_with_an_arrow() {
        for arrows in [false, true] {
            let arm = |client: &mut Client| {
                if arrows {
                    give_arrows(client);
                }
            };
            let Some((client, _keys, _to_net, used, _dir)) =
                mine_then_use_on(stack(BOW, 0), 20, None, block("dirt"), arm)
            else {
                return;
            };
            let aborted = !digging(PlayerDiggingStatus::AbortDestroyBlock, &used).is_empty();
            assert_eq!(aborted, arrows, "arrows={arrows}: {used:?}");
            assert_eq!(client.controller.is_using_item(), arrows);
            assert_eq!(client.controller.hitting_block().is_none(), arrows);
        }
    }

    /// The BedWars case: a sword at the bed (or chest, fence...) being broken. The block takes
    /// that click (`onBlockActivated` is true on the client), so the use would never start:
    /// the dig is left alone, as in vanilla, and nothing is sent. Sneaking skips the
    /// activation, so the same press then does block, with the usual one-tick packets.
    ///
    /// Negative control: what B2's rule (the held item alone decides) would have done there.
    /// The real right-click after an ABORT is consumed by the bed (a targeted C08 and a swing,
    /// no use), so that rule threw the bed's progress away and still did not block.
    #[test]
    fn a_block_that_takes_the_click_keeps_the_dig_and_sneaking_blocks() {
        for name in ["bed", "chest", "fence", "crafting_table"] {
            let Some((client, _keys, _to_net, used, _dir)) =
                mine_then_use_on(stack(SWORD, 0), 20, None, block(name), |_| {})
            else {
                return;
            };
            // The dig goes on (its swing may go out); the press is dropped.
            assert_eq!(placements(&used), (0, 0), "{name}: {used:?}");
            assert!(
                digging(PlayerDiggingStatus::AbortDestroyBlock, &used).is_empty(),
                "{name}: {used:?}"
            );
            assert_eq!(client.controller.hitting_block(), Some(DIRT), "{name}");
            assert!(!client.controller.is_using_item(), "{name}");

            let sneak = |client: &mut Client| {
                client
                    .session
                    .as_mut()
                    .and_then(crate::session::Session::predictor_mut)
                    .expect("predictor")
                    .state
                    .sneaking = true;
            };
            let Some((client, _keys, _to_net, used, _dir)) =
                mine_then_use_on(stack(SWORD, 0), 20, None, block(name), sneak)
            else {
                return;
            };
            assert_eq!(
                digging(PlayerDiggingStatus::AbortDestroyBlock, &used),
                [DIRT],
                "{name}, sneaking: {used:?}"
            );
            assert_eq!(placements(&used), (1, 1), "{name}, sneaking: {used:?}");
            assert!(client.controller.is_using_item(), "{name}, sneaking");
        }

        // Negative control: abort by hand (B2's rule) and run the real right-click at the bed.
        let Some((mut client, _keys, mut to_net, _used, _dir)) =
            mine_then_use_on(stack(SWORD, 0), 20, None, block("bed"), |_| {})
        else {
            return;
        };
        let step = client.controller.reset_block_removing();
        client.apply_dig_step(step);
        client.right_click_mouse();
        let sent = drained(&mut to_net);
        assert_eq!(
            digging(PlayerDiggingStatus::AbortDestroyBlock, &sent),
            [DIRT]
        );
        assert_eq!(placements(&sent), (0, 1), "the bed takes it: {sent:?}");
        assert!(
            !client.controller.is_using_item(),
            "no block: the abort only lost the dig"
        );
    }

    /// `interact::block_right_click_consumes` (the read-only dry run) against the real
    /// `right_click_mouse` (the side-effecting cascade), block by block: every 1.8.9 block id at
    /// meta 0, plus the states whose answer differs (a jukebox with a record, a powered button,
    /// both door halves, a malformed upper half), for a sword standing and sneaking, without
    /// and with `allowEdit`; and the one use item with an `onItemUse`, a carrot, on farmland.
    /// "Consumed" in the real run is "no untargeted C08". A new activation in the driver that
    /// the dry run lacks fails here.
    #[test]
    fn the_dry_run_matches_the_real_right_click_for_every_block() {
        use crate::app::interact::{block_right_click_consumes, BlockHit, HitResult};
        let Some((mut client, _dir, mut to_net, _to_game)) = mining_client(stack(SWORD, 0), 20)
        else {
            return;
        };
        let below = mc_math::BlockPos::new(0, 62, 0);
        let hit = BlockHit {
            pos: DIRT,
            face: mc_math::Facing::Up,
            hit_vec: mc_math::DVec3::new(0.5, 64.0, 0.5),
        };
        let mut states: Vec<(mc_data::BlockState, Option<mc_data::BlockState>)> = (1..=197_u16)
            .map(|id| (mc_data::BlockState::new(mc_data::BlockId(id), 0), None))
            .collect();
        let door = |meta| mc_data::BlockState::new(mc_data::BlockId(64), meta);
        states.extend([
            (mc_data::BlockState::new(mc_data::BlockId(84), 1), None),
            (mc_data::BlockState::new(mc_data::BlockId(77), 8), None),
            (mc_data::BlockState::new(mc_data::BlockId(143), 8), None),
            (door(8), Some(door(0))),
            (door(8), Some(block("stone"))),
            (mc_data::BlockState::new(mc_data::BlockId(71), 8), None),
        ]);
        let mut checked = 0;
        let mut consumed_count = 0;
        let mut wrong = Vec::new();
        for (held, cases) in [
            (stack(SWORD, 0), states.clone()),
            (
                stack(391, 0),
                vec![(block("farmland"), None), (block("dirt"), None)],
            ),
        ] {
            for sneaking in [false, true] {
                for allow_edit in [true, false] {
                    for &(state, under) in &cases {
                        // A fresh stack each time: a planted carrot is used up.
                        client.player_container.set_slot_contents(
                            super::super::super::HOTBAR_SLOT_OFFSET,
                            Some(held.clone()),
                        );
                        client.refresh_hotbar();
                        {
                            let session = client.session.as_mut().expect("session");
                            let world = &mut session.state_mut().world;
                            world.set_block(DIRT.up(), mc_data::BlockState::AIR);
                            world.set_block(below, under.unwrap_or(mc_data::BlockState::AIR));
                            world.set_block(DIRT, state);
                            session.state_mut().local.abilities.allow_edit = allow_edit;
                            session.predictor_mut().expect("predictor").state.sneaking = sneaking;
                        }
                        let dry = {
                            let session = client.session.as_ref().expect("session");
                            block_right_click_consumes(
                                &session.state().world,
                                DIRT,
                                hit.face,
                                &held,
                                sneaking,
                                allow_edit,
                                false,
                            )
                        };
                        let _ = drained(&mut to_net);
                        client.hit = HitResult::Block(hit);
                        client.right_click_mouse();
                        let sent = drained(&mut to_net);
                        let real = placements_anywhere(&sent).0 == 0;
                        client.controller.stop_using_item();
                        // An air-like state never reaches the cascade (`material != air`).
                        let air = state.props().is_air();
                        if !air {
                            checked += 1;
                            consumed_count += usize::from(real);
                            if dry != real {
                                wrong.push(format!(
                                    "{:?} meta {} (under {under:?}) item {} sneaking \
                                     {sneaking} allowEdit {allow_edit}: dry {dry}, real {real}",
                                    state.block(),
                                    state.meta(),
                                    held.item_id
                                ));
                            }
                        }
                    }
                }
            }
        }
        assert!(
            wrong.is_empty(),
            "{} mismatches:\n{}",
            wrong.len(),
            wrong.join("\n")
        );
        assert!(checked > 700, "{checked}");
        // The sweep really exercises both answers.
        assert!(consumed_count > 60, "{consumed_count}");
    }

    /// A PvP click that lands on a block starts a dig too. A use press during its first ticks is
    /// dropped exactly as vanilla drops it (block-hitting near blocks stays vanilla); a use held
    /// on takes over once the dig is a held one, through vanilla's held-button repeat (a dropped
    /// press never arms `rightClickDelayTimer`). Both buttons pressed together on dirt with a
    /// sword: three ticks of plain digging, then ABORT and the block in the fourth.
    #[test]
    fn a_click_length_dig_keeps_vanillas_dropped_press() {
        let Some((mut client, _dir, mut to_net, _to_game)) = mining_client(stack(SWORD, 0), 20)
        else {
            return;
        };
        use_stops_mining(&mut client, None);
        let mut keys = client.settings.keys.clone();
        keys.set_state(Input::MOUSE_LEFT, true);
        keys.set_state(Input::MOUSE_RIGHT, true);
        let first = tick(&mut client, &mut keys, &mut to_net);
        assert_eq!(
            digging(PlayerDiggingStatus::StartDestroyBlock, &first),
            [DIRT],
            "{first:?}"
        );
        for n in 0..USE_STOPS_MINING_MIN_DIG_TICKS {
            let sent = if n == 0 {
                first.clone()
            } else {
                tick(&mut client, &mut keys, &mut to_net)
            };
            assert_eq!(placements(&sent), (0, 0), "tick {n}: dropped, {sent:?}");
            assert!(digging(PlayerDiggingStatus::AbortDestroyBlock, &sent).is_empty());
            assert!(!client.controller.is_using_item(), "tick {n}");
            assert_eq!(client.controller.hitting_block(), Some(DIRT));
        }
        let held = tick(&mut client, &mut keys, &mut to_net);
        assert_eq!(
            digging(PlayerDiggingStatus::AbortDestroyBlock, &held),
            [DIRT],
            "{held:?}"
        );
        assert_eq!(placements(&held), (1, 1), "{held:?}");
        assert!(client.controller.is_using_item());
    }

    /// Like [`placements`], for a right-click at any block: `(untargeted, targeted)`.
    fn placements_anywhere(packets: &[C2SPlay]) -> (usize, usize) {
        let mut out = (0, 0);
        for packet in packets {
            if let C2SPlay::PlayerBlockPlacement(place) = packet {
                if place.position == mc_math::BlockPos::new(-1, -1, -1) {
                    out.0 += 1;
                } else {
                    out.1 += 1;
                }
            }
        }
        out
    }
}
