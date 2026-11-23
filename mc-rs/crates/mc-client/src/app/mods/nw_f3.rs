//! FPS: HUD, text and mods overhead, client half.
//!
//! Every candidate here is pixel-identical and has an `MC_NW_F3_*` kill switch read once per
//! process. The measurements and the reasoning are in
//!
//! * [`Client::nw_f3_prepare_ui_list`]: the live UI list leaves out the edge-on side quads of
//!   flat GUI item icons (`MC_NW_F3_ITEM_SLIVERS=0` keeps them), see
//!   `mc_ui::widget::DrawList::enable_item_sliver_culling`.
//! * [`Client::nw_f3_draw_tab_list`]: the Tab list's rows (display names, teams, scores, sort
//!   order) are rebuilt only when the player list, the scoreboard, the font or the language
//!   changed, instead of every frame it is held (`MC_NW_F3_TAB=0` rebuilds them every frame, as
//!   before). Skins, the hat layer and Dinnerbone's flip are still read every frame.

use std::collections::HashMap;

use mc_ui::screen_stack::ScreenCtx;
use mc_ui::screens::tab_list::{sort_rows, TabObjective, TabRenderType, TabRow};
use mc_ui::widget::{DrawList, UiTexture};
use uuid::Uuid;

use super::super::Client;
use crate::sync::scoreboard;

/// `MC_NW_F3_TAB=0` turns the cached Tab rows off. Read once per process.
fn nw_f3_tab_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        std::env::var_os("MC_NW_F3_TAB").is_none_or(|value| value != "0")
    });
    *ON
}

/// The state on the client (`Client::nw_f3`).
#[derive(Debug, Default)]
pub(crate) struct F3State {
    tab: TabRowsCache,
}

/// The inputs the cached Tab rows were built from. Everything a row's display name, team, score
/// and sort position reads is behind one of these: the player list's revision moves on every
/// S38 change (ping, game mode, display name, add, remove) and on a reconnect's fresh list; the
/// scoreboard's on every team/objective/score change, including the zero rows the rebuild itself
/// creates through `value_from_objective` (so the frame after such a rebuild rebuilds once more,
/// exactly as the per-frame build kept calling it); the font's on any pack reload, which also
/// reloads translations; the language code on a language change.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TabRowsKey {
    players: u64,
    scoreboard: u64,
    font: u64,
    show_heads: bool,
}

#[derive(Debug, Default)]
struct TabRowsCache {
    key: Option<TabRowsKey>,
    language: String,
    /// The display-slot-0 objective's name and overlay description.
    objective: Option<(String, TabObjective)>,
    /// Sorted, as `sort_rows` leaves them.
    rows: Vec<TabRow>,
    /// Each row's profile UUID, in row order.
    uuids: Vec<Uuid>,
    /// Each row's last custom-skin key and the UI texture built from it, so an unchanged skin
    /// does not re-`format!` its texture name every frame.
    skins: Vec<Option<(String, UiTexture)>>,
    /// Bumped on every rebuild: the overlay's measurement revision.
    revision: u64,
}

impl Client {
    /// In `build_ui`, right after the list is taken and Solid merging is
    /// switched on: turn on this lane's list options for the live frame. A `bool` test per frame
    /// once they are on (the list keeps them across frames).
    pub(crate) fn nw_f3_prepare_ui_list(&self, list: &mut DrawList) {
        if !list.item_sliver_culling() && mc_ui::widget::draw::nw_f3_item_slivers_enabled() {
            list.enable_item_sliver_culling();
        }
    }

    /// In `draw_tab_list`, after the overlay's update and the "held" test: draw
    /// the Tab list from rows that are rebuilt only when their inputs change. Returns `false`
    /// (and does nothing) when `MC_NW_F3_TAB=0`, so the caller runs its per-frame build.
    ///
    /// The output is the per-frame build's, row for row: the rebuild is that code moved behind
    /// [`TabRowsKey`], and the per-frame part (skins, the hat layer, the Dinnerbone flip, the
    /// skin uploads, header and footer) is the same code run every frame.
    pub(crate) fn nw_f3_draw_tab_list(&mut self, d: &mut DrawList, ctx: &ScreenCtx) -> bool {
        if !nw_f3_tab_enabled() {
            return false;
        }
        let Some(session) = &mut self.session else {
            return true;
        };
        let show_heads = session.login().is_some_and(|login| login.encrypted);
        let local_entity_present = session.predictor().is_some();
        let state = session.state_mut();
        let lang = &self.assets.language;
        let cache = &mut self.nw_f3.tab;
        let key = TabRowsKey {
            players: state.players.revision(),
            scoreboard: state.scoreboard.revision(),
            font: self.font.layout_generation(),
            show_heads,
        };
        if cache.key != Some(key) || cache.language != lang.code() {
            cache.key = Some(key);
            cache.language.clear();
            cache.language.push_str(lang.code());
            cache.revision = cache.revision.wrapping_add(1);
            // java: `GuiIngame.renderGameOverlay:346` passes `getObjectiveInDisplaySlot(0)` in,
            // which is what puts a score column beside every name.
            cache.objective = state
                .scoreboard
                .objective_in_display_slot(scoreboard::SLOT_LIST)
                .map(|objective| {
                    (
                        objective.name.clone(),
                        TabObjective {
                            display_name: objective.display_name.clone(),
                            render_type: match objective.render_type {
                                mc_protocol::play::s2c::ScoreboardRenderType::Hearts => {
                                    TabRenderType::Hearts
                                }
                                mc_protocol::play::s2c::ScoreboardRenderType::Integer => {
                                    TabRenderType::Integer
                                }
                            },
                        },
                    )
                });
            cache.rows.clear();
            for info in state.players.iter() {
                let mut row = TabRow::new(info.username.clone(), info.ping_ms);
                row.display = match &info.display_name {
                    Some(display) => display.formatted_text(lang),
                    None => state.scoreboard.format_player_name(&info.username),
                };
                row.team = state
                    .scoreboard
                    .player_team(&info.username)
                    .map(|team| team.name.clone())
                    .unwrap_or_default();
                row.spectator = info.game_mode == mc_protocol::play::s2c::GameMode::Spectator;
                if let Some((name, _)) = &cache.objective {
                    // java: `getValueFromObjective` **creates** a zeroed score rather than
                    // returning null, so an unscored player shows `0`, not a blank column.
                    row.score = state.scoreboard.value_from_objective(&info.username, name);
                }
                cache.rows.push(row);
            }
            sort_rows(&mut cache.rows);
            // Profile names are unique keys of the player list, so the sorted rows map back to
            // their UUIDs exactly.
            let uuid_by_name: HashMap<&str, Uuid> = state
                .players
                .iter()
                .map(|info| (info.username.as_str(), info.uuid))
                .collect();
            cache.uuids.clear();
            cache.uuids.extend(cache.rows.iter().map(|row| {
                uuid_by_name
                    .get(row.username.as_str())
                    .copied()
                    .unwrap_or_default()
            }));
            cache.skins.clear();
            cache.skins.resize(cache.rows.len(), None);
        }

        if show_heads {
            // java: `NetworkPlayerInfo.getLocationSkin` returns the default UUID skin until the
            // signed profile download is ready; `isWearing(HAT)` and the Dinnerbone check read
            // the live entity. Both can change without a list or scoreboard change, so they are
            // read every frame, exactly as the per-frame build does (#27: one entity pass).
            let mut entity_by_uuid: HashMap<Uuid, &crate::sync::entities::TrackedEntity> =
                HashMap::with_capacity(state.players.len());
            for (_, tracked) in state.entities.iter() {
                if let Some(uuid) = tracked.uuid {
                    entity_by_uuid.entry(uuid).or_insert(tracked);
                }
            }
            for ((row, &uuid), skin) in cache
                .rows
                .iter_mut()
                .zip(&cache.uuids)
                .zip(cache.skins.iter_mut())
            {
                let (model, custom_skin) = crate::app::appearance::player_skin_for_ui(
                    &mut self.appearance,
                    &state.players,
                    uuid,
                );
                if let Some((cache_key, rgba)) = custom_skin {
                    let texture = match skin {
                        Some((known, texture)) if *known == cache_key => texture.clone(),
                        _ => {
                            let texture = Self::ui_skin_texture_key(&cache_key);
                            *skin = Some((cache_key, texture.clone()));
                            texture
                        }
                    };
                    self.tab_skin_uploads.insert(texture.clone(), rgba);
                    row.skin = Some(texture);
                } else {
                    let path = model.default_texture();
                    // `UiTexture::gui(path)` is `minecraft:<path>`, stored verbatim.
                    let current = matches!(
                        &row.skin,
                        Some(UiTexture::Resource(location))
                            if location.namespace == "minecraft" && location.path == path
                    );
                    if !current {
                        row.skin = Some(UiTexture::gui(path));
                    }
                }
                let tracked_entity = entity_by_uuid.get(&uuid).copied();
                let skin_parts = if uuid == state.local.uuid && local_entity_present {
                    super::super::displayed_skin_parts(&self.settings)
                } else {
                    tracked_entity
                        .map(|tracked| {
                            crate::app::appearance::skin_parts_mask(&state.world, tracked) as u8
                        })
                        .unwrap_or(0)
                };
                row.hat = skin_parts & (1 << 6) != 0;
                row.upside_down = matches!(row.username.as_str(), "Dinnerbone" | "Grumm")
                    && skin_parts & 1 != 0
                    && ((uuid == state.local.uuid && local_entity_present)
                        || tracked_entity.is_some());
            }
        }

        let header = state
            .players
            .header
            .as_ref()
            .map(|c| c.formatted_text(lang));
        let footer = state
            .players
            .footer
            .as_ref()
            .map(|c| c.formatted_text(lang));
        self.tab_list.set_measure_revision(Some(cache.revision));
        self.tab_list.draw(
            d,
            &mut self.font,
            ctx.width,
            &cache.rows,
            header.as_deref(),
            footer.as_deref(),
            cache.objective.as_ref().map(|(_, o)| o),
            show_heads,
        );
        true
    }
}

#[cfg(test)]
mod tests {
    use mc_protocol::play::s2c::{
        AddPlayerEntry, DisplayScoreboard, GameMode, PlayerListItem, ScoreboardObjective,
        ScoreboardObjectiveMode, ScoreboardRenderType, TeamInfo, Teams, TeamsAction, UpdateScore,
        UpdateScoreAction,
    };

    use super::*;
    use crate::app::config::{ClientConfig, LoginMode};
    use crate::sync::event::ScoreboardEvent;

    fn offline_client() -> Option<(Client, tempfile::TempDir)> {
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

    fn team(name: &str, prefix: &str, players: &[&str]) -> ScoreboardEvent {
        ScoreboardEvent::Teams(Box::new(Teams {
            team_name: name.to_owned(),
            action: TeamsAction::Create {
                info: TeamInfo {
                    display_name: name.to_owned(),
                    prefix: prefix.to_owned(),
                    suffix: String::new(),
                    friendly_flags: 0,
                    name_tag_visibility: "always".to_owned(),
                    color: 12,
                },
                players: players.iter().map(|p| (*p).to_owned()).collect(),
            },
        }))
    }

    fn score(player: &str, value: i32) -> ScoreboardEvent {
        ScoreboardEvent::Score(UpdateScore {
            player_name: player.to_owned(),
            objective_name: "kills".to_owned(),
            action: UpdateScoreAction::Change { value },
        })
    }

    /// The cached rows draw exactly what the per-frame build draws, frame after frame and after
    /// every kind of change their key covers: latency, game mode, display name, team, score, a
    /// player leaving, the header.
    #[test]
    fn cached_tab_rows_draw_exactly_the_per_frame_rows() {
        let Some((mut client, _dir)) = offline_client() else {
            return;
        };
        let (handle, _to_game, _to_net) = crate::net::channel::NetHandle::new_pair();
        let mut session =
            crate::session::Session::new(handle, crate::session::JoinSettings::default());
        let add = |id: u128, name: &str, ping: i32, mode: GameMode, display: Option<&str>| {
            AddPlayerEntry {
                uuid: Uuid::from_u128(id),
                username: name.to_owned(),
                properties: Vec::new(),
                game_mode: mode,
                ping,
                display_name: display.map(str::to_owned),
            }
        };
        let state = session.state_mut();
        state
            .players
            .apply(&PlayerListItem::AddPlayer(vec![
                add(1, "zed", 40, GameMode::Survival, None),
                add(2, "ghost", 900, GameMode::Spectator, None),
                add(3, "alice", -1, GameMode::Creative, None),
                add(
                    4,
                    "Rival_E1",
                    83,
                    GameMode::Survival,
                    Some(r#"{"text":"Rival","color":"red"}"#),
                ),
                add(5, "bob", 155, GameMode::Survival, None),
                add(6, "Dinnerbone", 12, GameMode::Survival, None),
            ]))
            .expect("the rows must decode");
        state
            .players
            .set_header_footer(
                r#"{"text":"You are playing on MC.EXAMPLE.NET, a long header that wraps"}"#,
                r#"{"text":"store.example"}"#,
            )
            .expect("the header must decode");
        let board = &mut state.scoreboard;
        board.apply(&ScoreboardEvent::Objective(ScoreboardObjective {
            objective_name: "kills".to_owned(),
            mode: ScoreboardObjectiveMode::Create {
                display_name: "Kills".to_owned(),
                render_type: ScoreboardRenderType::Integer,
            },
        }));
        board.apply(&ScoreboardEvent::Display(DisplayScoreboard {
            position: 0,
            objective_name: "kills".to_owned(),
        }));
        board.apply(&team("red", "\u{a7}c[R] ", &["zed", "bob"]));
        board.apply(&team("blue", "\u{a7}9[B] ", &["alice"]));
        board.apply(&score("zed", 3));
        client.session = Some(session);
        client.tab_list_held = true;
        let ctx = ScreenCtx::new(client.resolution(854, 480), 0.0);

        let compare = |client: &mut Client, what: &str| {
            for frame in 0..3 {
                client.tab_list.set_measure_revision(None);
                let mut expected = DrawList::new();
                client.draw_tab_list_per_frame(&mut expected, &ctx);
                let mut got = DrawList::new();
                assert!(
                    client.nw_f3_draw_tab_list(&mut got, &ctx),
                    "the switch is on"
                );
                assert!(!got.is_empty(), "{what}: nothing drawn");
                assert!(
                    got == expected,
                    "{what}, frame {frame}: the cached rows differ"
                );
            }
        };
        compare(&mut client, "first frames");
        let session = client.session.as_mut().unwrap();
        session
            .state_mut()
            .players
            .apply(&PlayerListItem::UpdateLatency(vec![
                mc_protocol::play::s2c::UpdateLatencyEntry {
                    uuid: Uuid::from_u128(5),
                    ping: 1_200,
                },
            ]))
            .unwrap();
        compare(&mut client, "after a latency update");
        let session = client.session.as_mut().unwrap();
        session.state_mut().scoreboard.apply(&score("bob", 9));
        compare(&mut client, "after a score change");
        let session = client.session.as_mut().unwrap();
        session
            .state_mut()
            .scoreboard
            .apply(&team("green", "\u{a7}a[G] ", &["ghost"]));
        compare(&mut client, "after a team change");
        let session = client.session.as_mut().unwrap();
        session
            .state_mut()
            .players
            .apply(&PlayerListItem::RemovePlayer(vec![Uuid::from_u128(1)]))
            .unwrap();
        compare(&mut client, "after a player left");
        let session = client.session.as_mut().unwrap();
        session
            .state_mut()
            .players
            .set_header_footer(r#"{"text":"a new header"}"#, r#"{"text":""}"#)
            .unwrap();
        compare(&mut client, "after a header change");
    }
}
