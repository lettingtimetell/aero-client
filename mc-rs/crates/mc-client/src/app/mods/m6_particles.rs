//! The client half of the `particles` mod.
//!
//! The mod acts in three places, all configured from here when the settings generation moves:
//!
//! * the per-group amount and size table on the particle system
//!   ([`mc_render::particle::ParticleSystem::set_tuning`]), which every spawn path consults
//!   (`mc_render::particle::tuning`);
//! * the extra crit and sharpness emitters of hits you land, on the session
//!   (`Session::hit_particles`, asked once per attack in `attack_entity_with_knockback`);
//! * "Hide your potion swirls", applied in `Client::tick` where the local player's swirl
//!   colour is computed ([`Client::m6_own_swirl_colour`]).
//!
//! It also keeps the menu's group picker and its two sliders in step with the hidden per-group
//! `table` option (see `mc_mods::mods::particles`).
//!
//! With the mod off, a tick or frame costs one `u64` compare here, the particle system keeps no
//! table (one `Option` test per spawn attempt), and an attack costs one `bool` test.

use mc_mods::mods::particles::{self as catalog, GroupPercent, GROUP_COUNT};
use mc_render::particle::tuning::{GroupTuning, HitConfig, ParticleGroup};

use super::super::Client;

/// The settings-file id of the mod.
const MOD_ID: &str = "particles";

// The menu's table and the particle system's groups are the same seven, in the same order.
const _: () = assert!(GROUP_COUNT == ParticleGroup::COUNT);

/// What the client last read from the settings and pushed into the particle system.
#[derive(Debug)]
pub(crate) struct M6State {
    /// The settings generation last read. 0 = never (the store's generation starts at 1).
    seen: u64,
    /// The mod's on/off switch.
    enabled: bool,
    /// "Hide your potion swirls".
    hide_own_swirls: bool,
    /// What every session's hit emitters are set to ([`HitConfig::VANILLA`] while off).
    hits: HitConfig,
    /// The group whose row the two sliders show. `None` before the first read.
    shown_group: Option<usize>,
    /// Every group's values in whole percent (the parsed `table` option).
    table: [GroupPercent; GROUP_COUNT],
    /// Reused buffer for writing the `table` option.
    text: String,
}

impl Default for M6State {
    fn default() -> Self {
        M6State {
            seen: 0,
            enabled: false,
            hide_own_swirls: false,
            hits: HitConfig::VANILLA,
            shown_group: None,
            table: catalog::DEFAULT_TABLE,
            text: String::new(),
        }
    }
}

/// A whole-percent menu value as a multiplier (`100` is `1.0`).
fn multiplier(percent: u16) -> f32 {
    f32::from(percent) / 100.0
}

/// The particle system's per-group table for these menu values.
fn tuning_groups(table: &[GroupPercent; GROUP_COUNT]) -> [GroupTuning; ParticleGroup::COUNT] {
    std::array::from_fn(|i| GroupTuning {
        amount: multiplier(table[i].amount),
        size: multiplier(table[i].size),
    })
}

impl Client {
    /// The particles mod's hook.
    ///
    /// Off: one `u64` compare. On: that compare plus one small compare, because a reconnect
    /// builds a new session whose hit emitters start at vanilla.
    pub(crate) fn m6_sync(&mut self) {
        if self.m6.seen != self.mods.settings.generation() {
            self.m6_refresh();
        }
        if !self.m6.enabled {
            return;
        }
        if let Some(session) = self.session.as_mut() {
            if session.hit_particles.config() != self.m6.hits {
                session.hit_particles.set_config(self.m6.hits);
            }
        }
    }

    /// The local player's potion-swirl colour for this tick, or `0` (no
    /// swirl attempt at all) while "Hide your potion swirls" applies: the mod on, the option on,
    /// and the camera in first person. F5 and other players' swirls are untouched.
    pub(crate) fn m6_own_swirl_colour(&self, colour: i32) -> i32 {
        if self.m6.enabled && self.m6.hide_own_swirls && self.third_person_view == 0 {
            0
        } else {
            colour
        }
    }

    /// Read the settings after any change and push them into the particle system and the
    /// session.
    fn m6_refresh(&mut self) {
        self.m6_bind_picker();
        let settings = &self.mods.settings;
        let enabled = settings.enabled(MOD_ID);
        self.m6.enabled = enabled;
        self.m6.hide_own_swirls = settings.bool(MOD_ID, "hide_own_swirls");
        let table = &self.m6.table;
        self.m6.hits = if enabled {
            HitConfig {
                crit: multiplier(table[ParticleGroup::Crit.index()].amount),
                sharp: multiplier(table[ParticleGroup::Sharp.index()].amount),
                always_crit: settings.bool(MOD_ID, "always_crit"),
                always_sharp: settings.bool(MOD_ID, "always_sharp"),
            }
        } else {
            HitConfig::VANILLA
        };
        // `set_tuning` drops the table again when the values change nothing at the funnel (the
        // defaults: crits and sharpness above 100% only add emitters per hit).
        self.particles
            .set_tuning(enabled.then(|| tuning_groups(&self.m6.table)));
        // "Use particle physics". Collisions stay on (vanilla)
        // unless the mod is on and the option is off.
        self.particles
            .set_physics(!enabled || settings.bool(MOD_ID, "particle_physics"));
        if let Some(session) = self.session.as_mut() {
            session.hit_particles.set_config(self.m6.hits);
        }
        // The picker binding may have written options; the next call is a plain compare again.
        self.m6.seen = self.mods.settings.generation();
    }

    /// Keep the menu's group picker, its two sliders and the per-group `table` in step. Runs on
    /// every settings change, with the mod on or off, so the menu always shows the picked
    /// group's values:
    ///
    /// * another group was picked: load its row into the sliders;
    /// * otherwise, when the sliders differ from the picked group's row (a slider moved, or on
    ///   the first read the sliders were saved after the table), write them into the row.
    ///
    /// Compared in whole percent, so float noise never rewrites anything.
    fn m6_bind_picker(&mut self) {
        let settings = &mut self.mods.settings;
        let group = settings.choice(MOD_ID, "group").min(GROUP_COUNT - 1);
        self.m6.table = catalog::parse_table(settings.text(MOD_ID, "table"));
        let picked_another = self.m6.shown_group.is_some_and(|shown| shown != group);
        self.m6.shown_group = Some(group);
        if picked_another {
            let row = self.m6.table[group];
            settings.set_f32(MOD_ID, "amount", f32::from(row.amount));
            settings.set_f32(MOD_ID, "size", f32::from(row.size));
            return;
        }
        let sliders = GroupPercent::from_sliders(
            settings.f32(MOD_ID, "amount"),
            settings.f32(MOD_ID, "size"),
        );
        if sliders != self.m6.table[group] {
            self.m6.table[group] = sliders;
            catalog::write_table(&self.m6.table, &mut self.m6.text);
            settings.set_text(MOD_ID, "table", &self.m6.text);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::config::{ClientConfig, LoginMode};
    use crate::session::{JoinSettings, Session};

    /// A headless client over the real assets, its game dir a temp dir (so the mod settings
    /// file can never land in the user's `mc-rs`). `None` without the vanilla jar.
    fn headless_client() -> Option<(Client, tempfile::TempDir)> {
        let dir = tempfile::tempdir().expect("tempdir");
        let config = ClientConfig {
            game_dir: dir.path().to_path_buf(),
            login: LoginMode::Offline("Dev".to_owned()),
            ..ClientConfig::default()
        }
        .with_discovered_assets();
        if !config.version_jar.as_ref().is_some_and(|p| p.is_file()) {
            eprintln!("skip: no vanilla jar in this checkout");
            return None;
        }
        let client = Client::new(config, &mut |_, _| {}).expect("the real assets must load");
        Some((client, dir))
    }

    /// A session with no socket behind it; the channel ends must stay alive with it.
    fn new_session() -> (Session, impl Sized) {
        let (handle, to_game, to_net) = crate::net::channel::NetHandle::new_pair();
        (
            Session::new(handle, JoinSettings::default()),
            (to_game, to_net),
        )
    }

    /// A session that has received `JoinGame` (so it has a local player to tick).
    fn joined_session() -> (Session, impl Sized) {
        let (handle, to_game, to_net) = crate::net::channel::NetHandle::new_pair();
        let mut session = Session::new(handle, JoinSettings::default());
        let join = mc_protocol::play::s2c::JoinGame {
            entity_id: 3,
            hardcore_mode: false,
            game_mode: 0,
            dimension: 0,
            difficulty: 2,
            max_players: 20,
            level_type: "default".into(),
            reduced_debug_info: false,
        };
        to_game
            .try_send(crate::net::ToGame::Packet(
                mc_protocol::play::S2CPlay::JoinGame(join),
            ))
            .expect("join channel");
        assert!(session
            .poll_all()
            .iter()
            .any(|event| matches!(event, crate::sync::ClientEvent::JoinedGame(_))));
        (session, (to_game, to_net))
    }

    /// Whether the local player's potion swirl spawns within `ticks` real client ticks.
    fn own_swirl_within(client: &mut Client, ticks: usize) -> bool {
        client.particles.clear();
        for _ in 0..ticks {
            client.tick(&mc_physics::state::MoveInput::default());
            let spell = |p: &mc_render::particle::Particle| {
                matches!(p.kind, mc_render::particle::ParticleKind::Spell { .. })
            };
            if client.particles.particles().iter().any(spell) {
                return true;
            }
        }
        false
    }

    /// Through the live `Client::tick` (its `<<MOD-M6:tick>>` sync and the own-swirls hook): the
    /// local player's swirl is hidden in first person while the mod and the option are on, and
    /// kept in F5, with the option off and with the mod off.
    #[test]
    fn own_potion_swirls_hide_in_first_person_through_the_live_tick() {
        let Some((mut client, _dir)) = headless_client() else {
            return;
        };
        let (session, _channels) = joined_session();
        client.session = Some(session);
        client.ui = crate::app::driver::Ui::InGame;
        client.particles = mc_render::particle::ParticleSystem::with_seed(0);
        // Speed, with particles: a visible swirl attempt every tick (a coin flip each).
        let speed = crate::app::driver::TrackedEffect::from_packet(0, 2_000, true);
        client.active_effects.insert(1, speed);

        assert!(
            own_swirl_within(&mut client, 64),
            "mod off: vanilla's own swirl"
        );
        client.mods.settings.set_enabled(MOD_ID, true);
        assert!(!own_swirl_within(&mut client, 64), "hidden in first person");
        client.third_person_view = 1;
        assert!(own_swirl_within(&mut client, 64), "kept in F5");
        client.third_person_view = 0;
        client
            .mods
            .settings
            .set_bool(MOD_ID, "hide_own_swirls", false);
        assert!(
            own_swirl_within(&mut client, 64),
            "the option off keeps them"
        );
        client
            .mods
            .settings
            .set_bool(MOD_ID, "hide_own_swirls", true);
        assert!(!own_swirl_within(&mut client, 64));
        client.mods.settings.set_enabled(MOD_ID, false);
        assert!(own_swirl_within(&mut client, 64), "mod off again");
    }

    fn hits(client: &Client) -> HitConfig {
        client
            .session
            .as_ref()
            .expect("a session")
            .hit_particles
            .config()
    }

    #[test]
    fn settings_reach_the_particle_system_the_session_and_the_menu_picker() {
        let Some((mut client, _dir)) = headless_client() else {
            return;
        };
        let (session, _first) = new_session();
        client.session = Some(session);
        let colour = 0x0012_3456;

        // Off (the default): no table, vanilla hits, own swirls kept.
        client.m6_sync();
        assert!(client.particles.tuning().is_none());
        assert_eq!(hits(&client), HitConfig::VANILLA);
        assert_eq!(client.m6_own_swirl_colour(colour), colour);

        // On at the defaults: still no funnel table (crits and sharpness above 100% are
        // emitters per hit), 2x hits with always-sharpness, own swirls hidden in first person.
        client.mods.settings.set_enabled(MOD_ID, true);
        client.m6_sync();
        assert!(client.particles.tuning().is_none());
        let defaults = HitConfig {
            crit: 2.0,
            sharp: 2.0,
            always_crit: false,
            always_sharp: true,
        };
        assert_eq!(hits(&client), defaults);
        assert_eq!(client.m6_own_swirl_colour(colour), 0);
        client.third_person_view = 1;
        assert_eq!(client.m6_own_swirl_colour(colour), colour, "F5 keeps them");
        client.third_person_view = 0;

        // A reconnect builds a fresh session with vanilla hits; the next sync configures it.
        let (session, _second) = new_session();
        client.session = Some(session);
        assert_eq!(hits(&client), HitConfig::VANILLA);
        client.m6_sync();
        assert_eq!(hits(&client), defaults);

        // Pick "Lava": the sliders load its row.
        let settings = |client: &Client| {
            let s = &client.mods.settings;
            (s.f32(MOD_ID, "amount"), s.f32(MOD_ID, "size"))
        };
        client
            .mods
            .settings
            .set_choice(MOD_ID, "group", ParticleGroup::Lava.index());
        client.m6_sync();
        assert_eq!(settings(&client), (100.0, 100.0));
        // Move both sliders: the row and the particle system's table follow.
        client.mods.settings.set_f32(MOD_ID, "amount", 300.0);
        client.mods.settings.set_f32(MOD_ID, "size", 150.0);
        client.m6_sync();
        let table = client.mods.settings.text(MOD_ID, "table").to_owned();
        assert!(table.contains("lava=300/150"), "{table}");
        let lava = client
            .particles
            .tuning()
            .expect("a table")
            .group(ParticleGroup::Lava);
        assert_eq!(
            lava,
            GroupTuning {
                amount: 3.0,
                size: 1.5
            }
        );
        // Back to "Crits": its row returns, and Lava keeps its values.
        client
            .mods
            .settings
            .set_choice(MOD_ID, "group", ParticleGroup::Crit.index());
        client.m6_sync();
        assert_eq!(settings(&client), (200.0, 100.0));
        assert!(client
            .mods
            .settings
            .text(MOD_ID, "table")
            .contains("lava=300/150"));
        // Crits at 400% become four emitters per hit.
        client.mods.settings.set_f32(MOD_ID, "amount", 400.0);
        client.m6_sync();
        assert_eq!(hits(&client).crit, 4.0);
        // Settled: a sync writes nothing more.
        let generation = client.mods.settings.generation();
        client.m6_sync();
        client.m6_sync();
        assert_eq!(client.mods.settings.generation(), generation);

        // Off again: vanilla everywhere, the per-group values kept for next time.
        client.mods.settings.set_enabled(MOD_ID, false);
        client.m6_sync();
        assert!(client.particles.tuning().is_none());
        assert_eq!(hits(&client), HitConfig::VANILLA);
        assert!(!client
            .session
            .as_ref()
            .expect("a session")
            .hit_particles
            .is_active());
        assert_eq!(client.m6_own_swirl_colour(colour), colour);
        assert!(client
            .mods
            .settings
            .text(MOD_ID, "table")
            .contains("lava=300/150"));
        assert!(client
            .mods
            .settings
            .text(MOD_ID, "table")
            .contains("crit=400/100"));
    }

    /// "Use particle physics" reaches the particle system only while the mod is on;
    /// the default (on) and the mod off both keep vanilla's collisions.
    #[test]
    fn particle_physics_reaches_the_particle_system_only_with_the_mod_on() {
        let Some((mut client, _dir)) = headless_client() else {
            return;
        };
        client.m6_sync();
        assert!(client.particles.physics(), "mod off: vanilla");
        client.mods.settings.set_enabled(MOD_ID, true);
        client.m6_sync();
        assert!(client.particles.physics(), "the default is vanilla");
        client
            .mods
            .settings
            .set_bool(MOD_ID, "particle_physics", false);
        client.m6_sync();
        assert!(!client.particles.physics(), "switched off");
        client.mods.settings.set_enabled(MOD_ID, false);
        client.m6_sync();
        assert!(client.particles.physics(), "mod off again: vanilla");
        client.mods.settings.set_enabled(MOD_ID, true);
        client.m6_sync();
        assert!(
            !client.particles.physics(),
            "the option is kept for next time"
        );
        client
            .mods
            .settings
            .set_bool(MOD_ID, "particle_physics", true);
        client.m6_sync();
        assert!(client.particles.physics());
    }

    #[test]
    fn the_menu_table_and_the_particle_groups_share_keys_and_order() {
        for (i, group) in ParticleGroup::ALL.iter().enumerate() {
            assert_eq!(catalog::GROUP_KEYS[i], group.key(), "group {i}");
        }
    }

    #[test]
    fn menu_percentages_become_multipliers() {
        let mut table = catalog::DEFAULT_TABLE;
        table[ParticleGroup::Lava.index()] = GroupPercent {
            amount: 325,
            size: 50,
        };
        let groups = tuning_groups(&table);
        assert_eq!(
            groups[ParticleGroup::Lava.index()],
            GroupTuning {
                amount: 3.25,
                size: 0.5
            }
        );
        assert_eq!(groups[ParticleGroup::Crit.index()].amount, 2.0);
        assert_eq!(groups[ParticleGroup::Block.index()], GroupTuning::VANILLA);
    }
}
