//! Nametags' **Always show mob names** and **Confirm Disconnect**.
//!
//! # Always show mob names (`nametags` › `always_mob_names`, off by default)
//!
//! Vanilla 1.8.9 labels a mob's custom name only while the crosshair is on it:
//! `RenderLiving.canRenderName` is `super.canRenderName(entity) &&
//! (entity.getAlwaysRenderNameTagForRender() || entity.hasCustomName() && entity ==
//! renderManager.pointedEntity)`, and `EntityLivingBase.getAlwaysRenderNameTagForRender` returns
//! `false` (`EntityLiving` does not override it). The option takes the crosshair test away, so a
//! named mob is labelled the way a player is (`EntityPlayer.getAlwaysRenderNameTagForRender` is
//! `true`). Everything else is vanilla's own code path in `Client::upload_entities`:
//!
//! * **The gate** (`RendererLivingEntity.canRenderName`'s tail) still applies: F1 hides the label,
//!   an invisible mob gets none (unless you spectate, as vanilla's `isInvisibleToPlayer`), a mob
//!   with a rider gets none, and the entity you spectate gets none.
//! * **Range and sneaking:** the same row goes to the same `Nameplate::in_range` (64 blocks, 32
//!   while the mob sneaks) and the same frustum and render-distance gates as every label.
//! * **Players** are untouched: they already show their names, and a team that hides them is
//!   the server's choice (hide-and-seek games, NPCs built from players).
//! * **Armor stands** keep `ArmorStandRenderer.canRenderName` (CustomNameVisible alone): server
//!   holograms show exactly as before, and a nameless or hidden-name stand stays hidden.
//! * **No doubles** ([`MobNameLabels::drop_hologram_doubles`]): servers often float a hologram
//!   stand over a mob. When the stand rides the mob, vanilla's rider gate already hides the mob's
//!   own label. When the stand only follows it, the mob's label is dropped if a shown hologram
//!   sits on the label's spot (it would print over it) or floats just above the mob and already
//!   says the mob's name. A label vanilla itself would draw (the crosshair one) is never dropped.
//!
//! Cost: off, one `bool` per frame and one per label; on, two `Vec<usize>` kept between frames
//! (no allocation once warm) and, only when a frame has both a forced label and a hologram, a
//! distance test per pair.
//!
//! # Confirm Disconnect (`smart_disconnect`, off by default)
//!
//! Two ways out of a server can be clicked by mistake, and both ask while the mod is on:
//!
//! * **The pause menu's Disconnect.** The question is `mc_ui::screens::pause::PauseScreen`'s;
//!   [`confirm_disconnect`] is the one rule the client hands it before every click, and
//!   `Client::tick` counts its arm time down. Its Disconnect returns the unchanged
//!   `Action::Disconnect`, so leaving is byte for byte the pause menu's own teardown.
//! * **Joining another server from the pause menu's server list** (a double click on a row is
//!   enough). `Action::Connect` asks first ([`Client::fb_asks_before_join`]) with the same
//!   question over the list (`Ui::ConfirmLeaveServer`); "Join Server" sends the held
//!   `Action::Connect` back through the unchanged join path, once; Cancel and Escape return to
//!   the list with nothing joined.
//!
//! Nothing is sent while either question is up: the game keeps running as it does under the
//! pause menu and the list. A replay never asks.

use mc_math::Vec3;
use mc_mods::ModSettings;
use mc_ui::screens::multiplayer::ServerEntry;
use mc_ui::screens::yes_no::YesNoScreen;
use mc_ui::screens::Action;

use super::super::{Client, Ui};

/// One label row as `Client::upload_entities` collects it: the text, the camera-relative feet
/// it is anchored at, the squared distance, the entity height, child, sneaking, the scored
/// player and the label key.
pub(crate) type LabelRow = (String, Vec3, f32, f32, bool, bool, Option<String>, i32);

/// A hologram this close (horizontally) to a mob's label and this close in height prints over
/// it, whatever it says: blocks, and squared blocks.
const SAME_SPOT_RADIUS_SQ: f32 = 0.5 * 0.5;
const SAME_SPOT_HEIGHT: f32 = 0.3;
/// A hologram within this horizontal radius of the mob, from its feet up to this far above its
/// label, that says the mob's name, is the mob's nametag already.
const HOLOGRAM_RADIUS: f32 = 1.0;
const HOLOGRAM_RADIUS_SQ: f32 = HOLOGRAM_RADIUS * HOLOGRAM_RADIUS;
const HOLOGRAM_ABOVE: f32 = 2.5;

/// What kind of label a row is, for [`MobNameLabels`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LabelSource {
    /// A named mob's label that only "Always show mob names" shows (not under the crosshair).
    ForcedMob,
    /// An armor stand's label: a server hologram.
    Hologram,
    /// Anything else (players, the crosshair's mob, item frames, other entities).
    Other,
}

/// Which label rows came from where this frame. Kept between frames, so the option allocates
/// nothing once warm.
#[derive(Debug, Default)]
pub(crate) struct MobNameLabels {
    /// Rows labelled only because of the option, in row order.
    forced: Vec<usize>,
    /// Armor-stand rows, in row order.
    holograms: Vec<usize>,
}

impl MobNameLabels {
    /// Forget the last frame's rows.
    pub(crate) fn begin(&mut self) {
        self.forced.clear();
        self.holograms.clear();
    }

    /// Record that row `row` (the index it is pushed at) came from `source`.
    pub(crate) fn note(&mut self, row: usize, source: LabelSource) {
        match source {
            LabelSource::ForcedMob => self.forced.push(row),
            LabelSource::Hologram => self.holograms.push(row),
            LabelSource::Other => {}
        }
    }

    /// Drop each forced mob label that a shown hologram already covers (see the module docs).
    /// The order of the remaining rows is kept.
    pub(crate) fn drop_hologram_doubles(&mut self, rows: &mut Vec<LabelRow>) {
        if self.forced.is_empty() || self.holograms.is_empty() {
            return;
        }
        let holograms = &self.holograms;
        self.forced.retain(|&mob| {
            holograms
                .iter()
                .any(|&stand| hologram_covers(&rows[mob], &rows[stand]))
        });
        if self.forced.is_empty() {
            return;
        }
        // `forced` now holds the doubles, ascending.
        let doubles = &self.forced;
        let (mut row, mut next) = (0usize, 0usize);
        rows.retain(|_| {
            let drop = doubles.get(next) == Some(&row);
            next += usize::from(drop);
            row += 1;
            !drop
        });
    }
}

/// Which label a tracked entity's row is. `pointed`: the crosshair is on it.
pub(crate) fn label_source(
    tracked: &crate::sync::entities::TrackedEntity,
    pointed: bool,
) -> LabelSource {
    if tracked.uuid.is_some() || !tracked.kind.is_living() {
        return LabelSource::Other;
    }
    if tracked.kind.entity_type() == Some(mc_data::EntityType::ArmorStand) {
        LabelSource::Hologram
    } else if pointed {
        LabelSource::Other
    } else {
        LabelSource::ForcedMob
    }
}

/// Where a row's text is drawn: `renderLivingLabel`'s `y + height + 0.5`, with
/// `RendererLivingEntity.renderName`'s child drop (an armor stand's `isChild` is `isSmall`).
fn label_anchor(row: &LabelRow) -> Vec3 {
    let (_, feet, _, height, child, _, _, _) = row;
    let lift = if *child { height / 2.0 } else { *height };
    Vec3::new(feet.x, feet.y + lift + 0.5, feet.z)
}

/// Whether the hologram row `stand` already covers the forced mob row `mob`.
fn hologram_covers(mob: &LabelRow, stand: &LabelRow) -> bool {
    // Most pairs are far apart: two subtractions decide them.
    if (stand.1.x - mob.1.x).abs() > HOLOGRAM_RADIUS
        || (stand.1.z - mob.1.z).abs() > HOLOGRAM_RADIUS
    {
        return false;
    }
    let (at, hologram) = (label_anchor(mob), label_anchor(stand));
    let (dx, dz) = (hologram.x - at.x, hologram.z - at.z);
    let horizontal_sq = dx * dx + dz * dz;
    let dy = hologram.y - at.y;
    if horizontal_sq <= SAME_SPOT_RADIUS_SQ && dy.abs() <= SAME_SPOT_HEIGHT {
        return true;
    }
    let feet_below = at.y - mob.1.y;
    horizontal_sq <= HOLOGRAM_RADIUS_SQ
        && (-feet_below..=HOLOGRAM_ABOVE).contains(&dy)
        && contains_visible(&stand.0, &mob.0)
}

/// The characters a label shows: `§` and the code after it are dropped.
#[derive(Clone)]
struct Visible<'a>(std::str::Chars<'a>);

impl Iterator for Visible<'_> {
    type Item = char;

    fn next(&mut self) -> Option<char> {
        loop {
            match self.0.next()? {
                '\u{a7}' => {
                    self.0.next();
                }
                c => return Some(c),
            }
        }
    }
}

/// Whether `haystack` shows `needle`'s visible text (formatting codes ignored on both sides).
/// A needle that shows nothing is never contained. No allocation.
fn contains_visible(haystack: &str, needle: &str) -> bool {
    let needle = Visible(needle.chars());
    if needle.clone().next().is_none() {
        return false;
    }
    let mut start = Visible(haystack.chars());
    loop {
        let (mut h, mut n) = (start.clone(), needle.clone());
        loop {
            let Some(want) = n.next() else {
                return true;
            };
            match h.next() {
                Some(got) if got == want => {}
                Some(_) => break,
                None => return false,
            }
        }
        if start.next().is_none() {
            return false;
        }
    }
}

/// Whether the pause menu's Disconnect asks first: the mod is on
/// and there is a server to leave (closing a replay never asks).
pub(crate) fn confirm_disconnect(settings: &ModSettings, replaying: bool) -> bool {
    !replaying && settings.enabled(mc_mods::mods::smart_disconnect::ID)
}

/// The join question's second line.
fn join_detail(entry: &ServerEntry) -> String {
    let target = if entry.name.trim().is_empty() {
        entry.address.as_str()
    } else {
        entry.name.as_str()
    };
    format!("You will leave this server to join {target}.")
}

impl Client {
    /// `Action::Connect`'s guard: a join while a server is live (only
    /// the pause menu's server list and its Direct Connect screen can send one then) asks first
    /// when Confirm Disconnect is on. The join the question confirmed passes.
    pub(crate) fn fb_asks_before_join(&self) -> bool {
        !self.x3.fb_join_confirmed
            && self.session.is_some()
            && confirm_disconnect(&self.mods.settings, self.tw_play.playback.is_some())
    }

    /// Hold `entry` behind the question, drawn over the screen the join came from. Its buttons
    /// wake after `ARM_TICKS`, as the pause menu's question does.
    pub(crate) fn fb_ask_before_join(&mut self, entry: Box<ServerEntry>) {
        let language = &self.assets.language;
        let mut question = YesNoScreen::new(
            mc_ui::screens::pause::CONFIRM_DISCONNECT_TITLE,
            join_detail(&entry),
            language.tr("selectServer.select"),
            language.tr("gui.cancel"),
            0,
        )
        .with_world_loaded(true);
        question.set_button_delay(mc_mods::mods::smart_disconnect::ARM_TICKS);
        // As the delete question does: the list's pings stop while it is covered.
        if matches!(self.ui, Ui::Multiplayer(_)) {
            self.cancel_server_pings();
        }
        let parent = std::mem::replace(&mut self.ui, Ui::Loading);
        self.ui = Ui::ConfirmLeaveServer(Box::new(question), Box::new(parent), entry);
    }

    /// Close the join question: the screen it covered comes back; the held join is returned.
    pub(crate) fn fb_close_join_question(&mut self) -> Option<Box<ServerEntry>> {
        match std::mem::replace(&mut self.ui, Ui::Loading) {
            Ui::ConfirmLeaveServer(_, parent, entry) => {
                self.ui = *parent;
                Some(entry)
            }
            other => {
                self.ui = other;
                None
            }
        }
    }

    /// After a click: act on the join question's answer, once. "Join Server" restores the screen
    /// the join came from and sends the held `Action::Connect` through the unchanged join path.
    pub(crate) fn fb_drain_join_question(&mut self) {
        let answer = match &mut self.ui {
            Ui::ConfirmLeaveServer(question, _, _) => question.take_result(),
            _ => None,
        };
        let Some(answer) = answer else {
            return;
        };
        let entry = self.fb_close_join_question();
        if let (true, Some(entry)) = (answer.confirmed, entry) {
            self.x3.fb_join_confirmed = true;
            self.handle_action(Action::Connect(entry));
            self.x3.fb_join_confirmed = false;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(text: &str, feet: (f32, f32, f32), height: f32, child: bool) -> LabelRow {
        (
            text.to_owned(),
            Vec3::new(feet.0, feet.1, feet.2),
            100.0,
            height,
            child,
            false,
            None,
            0,
        )
    }

    const ZOMBIE: f32 = 1.95;
    const STAND: f32 = 1.975;

    #[test]
    fn visible_text_ignores_formatting_codes() {
        assert!(contains_visible(
            "\u{a7}c[Lv5] \u{a7}fZombie \u{a7}a100\u{a7}c\u{2764}\u{a7}r",
            "Zombie\u{a7}r"
        ));
        assert!(contains_visible("Bob\u{a7}r", "\u{a7}eB\u{a7}lob"));
        assert!(!contains_visible("Zombi", "Zombie"));
        assert!(!contains_visible("Zombie", "zombie"), "case counts");
        assert!(
            !contains_visible("anything", "\u{a7}r"),
            "a name that shows nothing never matches"
        );
        assert!(contains_visible("aab", "ab"), "a partial match restarts");
        assert!(contains_visible("x", "x"));
        assert!(!contains_visible("", "x"));
    }

    /// The doubles: a hologram on the mob's label spot (whatever it says), or above the mob
    /// saying its name. Kept: a crosshair label (not forced), a hologram saying something else
    /// above the mob, the same text on a hologram far away, and every other row.
    #[test]
    fn forced_labels_under_a_hologram_that_says_their_name_are_dropped() {
        let mut rows = vec![
            // 0: forced zombie "Guard" at the origin.
            row("Guard\u{a7}r", (0.0, 64.0, 0.0), ZOMBIE, false),
            // 1: a hologram stand standing on the zombie: its label prints at the same spot.
            row("Click me\u{a7}r", (0.0, 63.975, 0.0), STAND, false),
            // 2: forced zombie "Bob" at x=10.
            row("Bob\u{a7}r", (10.0, 64.0, 0.0), ZOMBIE, false),
            // 3: a hologram 1.2 blocks above Bob's label that says "Bob's pet".
            row("\u{a7}eBob's pet\u{a7}r", (10.2, 65.2, 0.0), STAND, false),
            // 4: forced zombie "Alice" at x=20, with a hologram above that says something else.
            row("Alice\u{a7}r", (20.0, 64.0, 0.0), ZOMBIE, false),
            row(
                "\u{2764}\u{2764}\u{2764}\u{a7}r",
                (20.0, 65.0, 0.0),
                STAND,
                false,
            ),
            // 6: forced zombie "Carl" at x=30; a "Carl" hologram 5 blocks away.
            row("Carl\u{a7}r", (30.0, 64.0, 0.0), ZOMBIE, false),
            row("Carl\u{a7}r", (35.0, 64.0, 0.0), STAND, false),
            // 8: a crosshair label (not forced) under a same-name hologram: vanilla's, kept.
            row("Dora\u{a7}r", (40.0, 64.0, 0.0), ZOMBIE, false),
            row("Dora\u{a7}r", (40.0, 65.0, 0.0), STAND, false),
            // 10: a player.
            row("Steve\u{a7}r", (0.0, 64.0, 5.0), 1.8, false),
        ];
        let mut labels = MobNameLabels::default();
        labels.begin();
        let sources = [
            LabelSource::ForcedMob,
            LabelSource::Hologram,
            LabelSource::ForcedMob,
            LabelSource::Hologram,
            LabelSource::ForcedMob,
            LabelSource::Hologram,
            LabelSource::ForcedMob,
            LabelSource::Hologram,
            LabelSource::Other,
            LabelSource::Hologram,
            LabelSource::Other,
        ];
        for (index, source) in sources.into_iter().enumerate() {
            labels.note(index, source);
        }
        labels.drop_hologram_doubles(&mut rows);
        let texts: Vec<&str> = rows.iter().map(|row| row.0.as_str()).collect();
        assert_eq!(
            texts,
            [
                "Click me\u{a7}r",
                "\u{a7}eBob's pet\u{a7}r",
                "Alice\u{a7}r",
                "\u{2764}\u{2764}\u{2764}\u{a7}r",
                "Carl\u{a7}r",
                "Carl\u{a7}r",
                "Dora\u{a7}r",
                "Dora\u{a7}r",
                "Steve\u{a7}r",
            ],
            "Guard (same spot) and Bob (named above) dropped, in order"
        );
    }

    /// Nothing to compare: no work, no change.
    #[test]
    fn without_forced_labels_or_holograms_nothing_changes() {
        let original = vec![
            row("Guard\u{a7}r", (0.0, 64.0, 0.0), ZOMBIE, false),
            row("Guard\u{a7}r", (0.0, 63.975, 0.0), STAND, false),
        ];
        for sources in [
            [LabelSource::Other, LabelSource::Hologram],
            [LabelSource::ForcedMob, LabelSource::Other],
        ] {
            let mut rows = original.clone();
            let mut labels = MobNameLabels::default();
            labels.begin();
            for (index, source) in sources.into_iter().enumerate() {
                labels.note(index, source);
            }
            labels.drop_hologram_doubles(&mut rows);
            assert_eq!(rows, original);
        }
        // `begin` forgets the last frame.
        let mut labels = MobNameLabels::default();
        labels.note(0, LabelSource::ForcedMob);
        labels.note(1, LabelSource::Hologram);
        labels.begin();
        let mut rows = original.clone();
        labels.drop_hologram_doubles(&mut rows);
        assert_eq!(rows, original);
    }

    /// A small stand (`isChild`) and a baby mob anchor half a height lower, as vanilla draws them.
    #[test]
    fn child_rows_anchor_half_a_height_lower() {
        let baby = row("Tiny\u{a7}r", (0.0, 64.0, 0.0), 0.975, true);
        let small_stand = row("Other\u{a7}r", (0.0, 64.0, 0.0), 0.9875, true);
        assert!((label_anchor(&baby).y - (64.0 + 0.4875 + 0.5)).abs() < 1e-5);
        assert!(
            hologram_covers(&baby, &small_stand),
            "both labels at the same spot"
        );
        let far_below = row("Tiny\u{a7}r", (0.0, 60.0, 0.0), STAND, false);
        assert!(
            !hologram_covers(&baby, &far_below),
            "a hologram under the feet"
        );
    }

    #[test]
    fn smart_disconnect_asks_only_with_a_server_to_leave() {
        let mut settings = ModSettings::new();
        assert!(!confirm_disconnect(&settings, false), "off by default");
        settings.set_enabled(mc_mods::mods::smart_disconnect::ID, true);
        assert!(confirm_disconnect(&settings, false));
        assert!(
            !confirm_disconnect(&settings, true),
            "a replay closes at once"
        );
    }

    // ---- Confirm Disconnect through the real client ----------------------------------------

    use super::super::super::{Client, Ui};
    use crate::app::config::{ClientConfig, LoginMode};
    use crate::app::state::Transition;
    use mc_mods::mods::smart_disconnect::{ARM_TICKS, ID as SMART_DISCONNECT};
    use mc_ui::screen_stack::ScreenCtx;

    /// A real offline client over a temporary game dir (skips without the vanilla jar).
    fn offline_client() -> Option<(Client, tempfile::TempDir)> {
        let dir = tempfile::tempdir().expect("tempdir");
        let config = ClientConfig {
            game_dir: dir.path().to_path_buf(),
            login: LoginMode::Offline("Dev".to_owned()),
            ..ClientConfig::default()
        }
        .with_discovered_assets();
        if !config.version_jar.as_ref().is_some_and(|p| p.is_file()) {
            eprintln!("skip: no vanilla jar in this checkout (set MC_JARS_DIR)");
            return None;
        }
        let client = Client::new(config, &mut |_, _| {}).expect("the real assets must load");
        Some((client, dir))
    }

    /// Put `client` in a world over a socketless session; the channel ends come back so the test
    /// can read what the client sends.
    fn join_fake_world(
        client: &mut Client,
    ) -> (
        tokio::sync::mpsc::Sender<crate::net::ToGame>,
        tokio::sync::mpsc::UnboundedReceiver<crate::net::ToNet>,
    ) {
        let (handle, to_game, to_net) = crate::net::channel::NetHandle::new_pair();
        client
            .machine
            .apply(Transition::Connect {
                host: "127.0.0.1".to_owned(),
                port: 25_565,
            })
            .expect("menu -> connecting");
        client
            .machine
            .apply(Transition::Joined)
            .expect("connecting -> in world");
        client.session = Some(crate::Session::new(
            handle,
            crate::session::JoinSettings::default(),
        ));
        client.ui = Ui::InGame;
        (to_game, to_net)
    }

    /// A left click at `(x, y)` (GUI pixels) through `Client::mouse_clicked`, after the frame's
    /// `init` has laid the pause menu out.
    fn click(client: &mut Client, ctx: ScreenCtx, x: i32, y: i32) {
        if let Ui::Pause(screen) = &mut client.ui {
            screen.init(&ctx, &client.assets.language);
        }
        client.cursor = (x, y);
        client.mouse_clicked(0, ctx);
    }

    fn pause_disconnect(ctx: &ScreenCtx) -> (i32, i32) {
        (ctx.width / 2 - 100 + 1, ctx.height / 4 + 104 + 1)
    }

    fn question_button(ctx: &ScreenCtx, confirm: bool) -> (i32, i32) {
        let x = if confirm {
            ctx.width / 2 - 155
        } else {
            ctx.width / 2 + 5
        };
        (x + 1, ctx.height / 6 + 96 + 1)
    }

    fn asking(client: &Client) -> bool {
        matches!(&client.ui, Ui::Pause(screen) if screen.is_confirming_disconnect())
    }

    fn sent(to_net: &mut tokio::sync::mpsc::UnboundedReceiver<crate::net::ToNet>) -> Vec<String> {
        std::iter::from_fn(|| to_net.try_recv().ok())
            .map(|message| format!("{message:?}"))
            .collect()
    }

    /// On: the pause menu's Disconnect asks and sends nothing; the client's tick arms the
    /// question; Cancel keeps the connection; the question's Disconnect is the button's own
    /// teardown (the channel closed, no packet, the server list shown). Off: one click leaves.
    #[test]
    fn smart_disconnect_through_the_real_pause_menu() {
        let Some((mut client, _dir)) = offline_client() else {
            return;
        };
        let (_to_game, mut to_net) = join_fake_world(&mut client);
        client.mods.settings.set_enabled(SMART_DISCONNECT, true);
        client.show_screen(Some(Ui::Pause(Box::default())));
        let ctx = ScreenCtx::new(client.resolution(854, 480), 0.0);

        let (x, y) = pause_disconnect(&ctx);
        click(&mut client, ctx, x, y);
        assert!(asking(&client), "the question is up");
        assert!(client.session.is_some() && client.machine.state().is_in_world());
        assert!(sent(&mut to_net).is_empty(), "asking sends nothing");
        // Too early: the question's Disconnect is not armed yet.
        let (qx, qy) = question_button(&ctx, true);
        click(&mut client, ctx, qx, qy);
        assert!(asking(&client) && client.session.is_some());

        // The client's tick arms it.
        let input = mc_physics::state::MoveInput::default();
        for _ in 0..ARM_TICKS {
            client.tick(&input);
        }
        assert!(
            matches!(&client.ui, Ui::Pause(screen) if screen.confirmation_armed()),
            "armed after {ARM_TICKS} client ticks"
        );
        let ticked = sent(&mut to_net);
        assert!(
            !ticked.iter().any(|message| message.contains("Disconnect")),
            "the question never closes the channel by itself: {ticked:?}"
        );

        // Cancel: the pause menu again, still connected.
        let (cx, cy) = question_button(&ctx, false);
        click(&mut client, ctx, cx, cy);
        assert!(matches!(client.ui, Ui::Pause(_)) && !asking(&client));
        assert!(client.session.is_some() && client.machine.state().is_in_world());
        assert!(sent(&mut to_net).is_empty(), "cancel sends nothing");

        // Ask again, wait, confirm: exactly the pause menu's own Disconnect.
        click(&mut client, ctx, x, y);
        assert!(asking(&client));
        for _ in 0..ARM_TICKS {
            client.tick(&input);
        }
        let _ = sent(&mut to_net);
        click(&mut client, ctx, qx, qy);
        let messages = sent(&mut to_net);
        assert_eq!(
            messages,
            ["Disconnect".to_owned()],
            "the channel is closed and no packet is sent"
        );
        assert!(client.session.is_none(), "the world is unloaded");
        assert!(
            matches!(client.ui, Ui::Multiplayer(_)),
            "the server list, as the pause menu's Disconnect shows it"
        );
        client.active_connect_resolution = None;

        // Off: vanilla, one click leaves.
        let (_to_game, mut to_net) = join_fake_world(&mut client);
        client.mods.settings.set_enabled(SMART_DISCONNECT, false);
        client.show_screen(Some(Ui::Pause(Box::default())));
        click(&mut client, ctx, x, y);
        assert_eq!(sent(&mut to_net), ["Disconnect".to_owned()]);
        assert!(client.session.is_none());
        assert!(matches!(client.ui, Ui::Multiplayer(_)));
    }

    fn escape() -> crate::app::input::KeyPress {
        crate::app::input::KeyPress {
            sequence: 0,
            code: winit::keyboard::KeyCode::Escape,
            lwjgl: mc_ui::widget::keys::ESCAPE,
            typed: None,
            mods: mc_ui::widget::keys::Modifiers::NONE,
            repeat: false,
        }
    }

    /// The other way out: joining another server from the pause menu's list. On: the join waits
    /// behind the question (still connected, nothing sent); Escape and Cancel return to the list;
    /// "Join Server", once the client's ticks armed it, runs the unchanged join (the old channel
    /// closed with no packet, then the connecting screen whose parent is the list). Off, the `joining_from_the_in_game_list_disconnects_cleanly_first` covers the vanilla path.
    #[test]
    fn smart_disconnect_asks_before_joining_another_server_from_the_in_game_list() {
        let Some((mut client, _dir)) = offline_client() else {
            return;
        };
        let (_to_game, mut to_net) = join_fake_world(&mut client);
        client.mods.settings.set_enabled(SMART_DISCONNECT, true);
        client.show_screen(Some(Ui::Pause(Box::default())));
        client.handle_action(Action::OpenMultiplayer);
        assert!(matches!(client.ui, Ui::Multiplayer(_)));
        // TEST-NET-1: never routable, and the test never polls the resolution anyway.
        let entry = || Box::new(ServerEntry::new("Test", "192.0.2.1:25565"));
        let ctx = ScreenCtx::new(client.resolution(854, 480), 0.0);
        let lay_out = |client: &mut Client| {
            if let Ui::ConfirmLeaveServer(question, _, _) = &mut client.ui {
                question.resize_keeping_delay(ctx.width, ctx.height);
            }
        };
        let input = mc_physics::state::MoveInput::default();

        client.handle_action(Action::Connect(entry()));
        assert!(
            matches!(client.ui, Ui::ConfirmLeaveServer(..)),
            "the join waits"
        );
        assert_eq!(client.ui.name(), "confirm_leave_server");
        assert!(client.session.is_some() && client.machine.state().is_in_world());
        assert!(sent(&mut to_net).is_empty(), "asking sends nothing");

        // Escape: back to the list, nothing joined.
        client.key_pressed(escape());
        assert!(
            matches!(client.ui, Ui::Multiplayer(_)),
            "Escape: the list again"
        );
        assert!(client.session.is_some());

        // Ask again: too early to answer, then armed by the client's ticks, then Cancel.
        client.handle_action(Action::Connect(entry()));
        lay_out(&mut client);
        let (jx, jy) = question_button(&ctx, true);
        click(&mut client, ctx, jx, jy);
        assert!(
            matches!(client.ui, Ui::ConfirmLeaveServer(..)),
            "not armed yet"
        );
        for _ in 0..ARM_TICKS {
            client.tick(&input);
        }
        lay_out(&mut client);
        let (cx, cy) = question_button(&ctx, false);
        click(&mut client, ctx, cx, cy);
        assert!(
            matches!(client.ui, Ui::Multiplayer(_)),
            "Cancel: the list again"
        );
        assert!(client.session.is_some() && client.machine.state().is_in_world());
        let messages = sent(&mut to_net);
        assert!(
            !messages
                .iter()
                .any(|message| message.contains("Disconnect")),
            "still connected: {messages:?}"
        );

        // Ask, arm, Join Server: the unchanged join, exactly once.
        client.handle_action(Action::Connect(entry()));
        for _ in 0..ARM_TICKS {
            client.tick(&input);
        }
        lay_out(&mut client);
        let _ = sent(&mut to_net);
        click(&mut client, ctx, jx, jy);
        assert_eq!(
            sent(&mut to_net),
            ["Disconnect".to_owned()],
            "the old channel is closed and no packet is sent"
        );
        assert!(client.session.is_none(), "the old world is gone");
        let Ui::Connecting(screen) = &client.ui else {
            panic!("the join continues on the connecting screen");
        };
        assert_eq!(
            screen.parent(),
            mc_ui::screens::connect::ConnectionParent::Multiplayer,
            "a failed join returns to the server list"
        );
        assert!(!client.x3.fb_join_confirmed, "the pass is spent");
        client.active_connect_resolution = None;
    }
}
