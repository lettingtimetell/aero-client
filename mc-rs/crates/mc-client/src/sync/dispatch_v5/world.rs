//! The 1.7.10 world and effect packets: block events, explosions, world events, sounds,
//! particles, game state, maps, signs and tile entities (java17
//! `client/network/NetHandlerPlayClient.java`, `NHPC` below).

use mc_math::BlockPos;
use mc_protocol::play::{s2c, S2CPlay};
use mc_protocol::v5::s2c as s2c5;
use mc_protocol::v5::s2c::MapUpdate;
use mc_protocol::ProtoError;
use mc_ui::font::component::ChatComponent;
use mc_world::tileent::TileKind;

use crate::error::ClientError;
use crate::sync::dispatch;
use crate::sync::event::{ChatPosition, ClientEvent};
use crate::sync::state::{game_mode_from_id, ClientState};

/// The side of a 1.7.10 map image (`MapData.colors` is 128 x 128).
const MAP_SIZE: usize = 128;

/// java17 `NHPC.handleBlockAction` (`:1166-1169`) = java18 (`:1322-1326`): `addBlockEvent` with
/// the packet's block (`Block.getBlockById(id & 4095)` in both decoders).
pub(super) fn lower_block_action(action: &s2c5::BlockAction) -> S2CPlay {
    S2CPlay::BlockAction(s2c::BlockAction {
        position: BlockPos::new(action.x, i32::from(action.y), action.z),
        data1: action.data1,
        data2: action.data2,
        block_id: action.block_id & 0xFFF,
    })
}

/// java17 `NHPC.handleBlockBreakAnim` (`:1171-1174`) = java18 (`:1331-1335`): progress 0-9 shows
/// that crack stage for the breaker, any other value (255 included) removes the breaker's
/// overlay (`RenderGlobal.destroyBlockPartially` in both).
pub(super) fn lower_break_animation(animation: &s2c5::BlockBreakAnimation) -> S2CPlay {
    S2CPlay::BlockBreakAnimation(s2c::BlockBreakAnimation {
        breaker_id: animation.breaker_id,
        position: BlockPos::new(animation.x, animation.y, animation.z),
        progress: animation.progress,
    })
}

/// java17 `NHPC.handleExplosion` (`:867-875`) = java18 (`:1078-1086`): the same affected-block
/// list (`(int)` origin plus signed byte offsets in both decoders), `doExplosionB(true)` and the
/// knockback added to the player's motion.
pub(super) fn lower_explosion(explosion: s2c5::Explosion) -> S2CPlay {
    S2CPlay::Explosion(s2c::Explosion {
        x: f64::from(explosion.x),
        y: f64::from(explosion.y),
        z: f64::from(explosion.z),
        strength: explosion.strength,
        affected_blocks: explosion
            .offsets
            .iter()
            .map(|&(dx, dy, dz)| s2c::ExplosionAffectedBlock { dx, dy, dz })
            .collect(),
        motion_x: explosion.motion_x,
        motion_y: explosion.motion_y,
        motion_z: explosion.motion_z,
    })
}

/// java17 `NHPC.handleEffect` (`:1267-1277`) dispatches as java18 (`:1441-1453`) does, into a
/// 1.7.10 world-event table (java17 `RenderGlobal.playAuxSFX`, `client/renderer/
/// RenderGlobal.java:2128-2372`) that differs in three ids: 1006 (java18's door-close sound) is
/// unknown to 1.7.10 and does nothing; 2001's block is `id & 4095` with meta `data >> 12 & 255`,
/// a 1.7.10 block that goes through the 1.7.10 block table; 2006 (fall dust) is 1.7.10's only,
/// and the client plays it, with 1000-1002's corner position, from its 1.7.10 table
/// (`app/effects_v5.rs`). `None`: the event does nothing in 1.7.10.
pub(super) fn lower_effect(effect: &s2c5::Effect) -> Option<S2CPlay> {
    let data = match effect.effect_id {
        1006 if !effect.disable_relative_volume => return None,
        2001 if !effect.disable_relative_volume => {
            mc_data::v5::state_from_v5_id_meta(effect.data & 4095, effect.data >> 12 & 255)
                .to_java_state_id()
        }
        _ => effect.data,
    };
    Some(S2CPlay::Effect(s2c::Effect {
        effect_id: effect.effect_id,
        position: BlockPos::new(effect.x, i32::from(effect.y), effect.z),
        data,
        disable_relative_volume: effect.disable_relative_volume,
    }))
}

/// java17 `NHPC.handleSoundEffect` (`:1400-1403`) = java18 (`:1695-1699`): `playSound` by name.
/// A name only 1.8.9's sound list has is an unknown event to 1.7.10, which drops it ("Unable to
/// play unknown soundEvent", java17 `SoundManager.playSound`); every other name, a resource
/// pack's own included, goes the 1.8.9 way. `None`: nothing is played.
pub(super) fn lower_sound(sound: s2c5::SoundEffect) -> Option<S2CPlay> {
    let only_1_8_9 = mc_data::v5::sound_event_v5(&sound.sound_name).is_none()
        && mc_data::sound_event(&sound.sound_name).is_some();
    if only_1_8_9 {
        return None;
    }
    Some(S2CPlay::SoundEffect(s2c::SoundEffect {
        sound_name: sound.sound_name,
        x_fixed: sound.x,
        y_fixed: sound.y,
        z_fixed: sound.z,
        volume: sound.volume,
        pitch_quantized: sound.pitch,
    }))
}

/// java17 `NHPC.handleParticles` (`:1580-1602`): the particle is named
/// ([`mc_data::v5::particle_from_v5_name`]). A name 1.7.10 does not know spawns nothing; a
/// prefixed name it cannot parse throws out of `RenderGlobal.doSpawnParticle` (a crash in
/// 1.7.10), reported here as an error. Otherwise the 1.8.9 particle with the same count, offset
/// and speed draws, where only `hugeexplosion`, `largeexplode` and `fireworksSpark` skip the
/// 16-block and particle-setting checks (java17 `RenderGlobal.doSpawnParticle:1840-1880`): the
/// packet's long-distance flag carries that.
pub(super) fn particles(
    particles: &s2c5::Particles,
    events: &mut Vec<ClientEvent>,
) -> Result<(), ClientError> {
    use mc_data::v5::ParticleV5;
    let particle = match mc_data::v5::particle_from_v5_name(&particles.particle_name) {
        ParticleV5::Spawn(particle) => particle,
        ParticleV5::Unknown => return Ok(()),
        ParticleV5::Malformed => {
            tracing::warn!(name = %particles.particle_name, "1.7.10 particle name it cannot parse");
            return Err(ClientError::Protocol(ProtoError::InvalidEnum {
                what: "particle name 1.7.10 cannot parse",
                value: 0,
            }));
        }
    };
    events.push(ClientEvent::Particles(Box::new(s2c::Particles {
        particle_id: particle.id,
        long_distance: particle.ignore_range,
        x: particles.x,
        y: particles.y,
        z: particles.z,
        offset_x: particles.offset_x,
        offset_y: particles.offset_y,
        offset_z: particles.offset_z,
        particle_speed: particles.speed,
        particle_count: particles.count,
        extra_data: particle.args().to_vec(),
    })));
    Ok(())
}

/// `WorldSettings.GameType.getByID` as 1.7.10 has it: `NOT_SET` (-1), survival, creative and
/// adventure, survival for anything else, 3 included (java17 `world/WorldSettings.java:98-101,
/// 158-172`; 1.8.9 adds spectator as 3).
fn game_mode_v5(id: i32) -> s2c::GameMode {
    match id {
        -1 | 1 | 2 => game_mode_from_id(id),
        _ => s2c::GameMode::Survival,
    }
}

/// java17 `NHPC.handleChangeGameState` (`:1195-1258`). The chat line comes first and is printed
/// for `MESSAGE_NAMES[reason]` (`{"tile.bed.notValid", null, null, "gameMode.changed"}`, java17
/// `S2BPacketChangeGameState.java:11`); reason 3 then sets 1.7.10's game type; reason 10 (java18's
/// elder-guardian effect) does not exist. Reasons 1, 2 and 4-8 are java18's (`:1358-1427`).
pub(super) fn change_game_state(
    state: &mut ClientState,
    game_state: &s2c5::ChangeGameState,
    events: &mut Vec<ClientEvent>,
) -> Result<(), ClientError> {
    match game_state.reason {
        3 => {
            events.push(ClientEvent::Chat {
                component: Box::new(ChatComponent::translate("gameMode.changed", vec![])),
                position: ChatPosition::Chat,
            });
            let mode = game_mode_v5(mc_math::floor_f32(game_state.value + 0.5));
            state.local.game_mode = mode;
            events.push(ClientEvent::GameModeChanged(mode));
            Ok(())
        }
        10 => Ok(()),
        reason => {
            use s2c::GameStateReason as R;
            let reason = match reason {
                0 => R::InvalidBed,
                1 => R::EndRaining,
                2 => R::BeginRaining,
                4 => R::EnterCredits,
                5 => R::DemoMessage,
                6 => R::ArrowHittingPlayer,
                7 => R::RainLevelChange,
                8 => R::ThunderLevelChange,
                other => R::Other(other),
            };
            let packet = s2c::ChangeGameState {
                reason,
                value: game_state.value,
            };
            dispatch::apply(state, S2CPlay::ChangeGameState(packet), events)
        }
    }
}

/// java17 `NHPC.handleUpdateSign` (`:1077-1107`) = java18 (`:1234-1261`), whose lines are chat
/// components where 1.7.10's are plain strings: each line becomes a text component holding the
/// string, which draws the same (formatting codes included).
pub(super) fn lower_update_sign(sign: s2c5::UpdateSign) -> S2CPlay {
    S2CPlay::UpdateSign(s2c::UpdateSign {
        position: BlockPos::new(sign.x, i32::from(sign.y), sign.z),
        lines: sign.lines.map(|line| ChatComponent::text(line).to_json()),
    })
}

/// java17 `NHPC.handleUpdateTileEntity` (`:1109-1139`): actions 1-5 (spawner, command block,
/// beacon, skull, flower pot), each read into a tile entity of that class, as java18
/// (`:1267-1281`) does; 1.7.10 has no 6 (banner) and ignores every other action. `None`: ignored.
pub(super) fn lower_tile_entity(tile: s2c5::UpdateTileEntity) -> Option<S2CPlay> {
    (1..=5).contains(&tile.action).then(|| {
        S2CPlay::UpdateTileEntity(s2c::UpdateTileEntity {
            position: BlockPos::new(tile.x, i32::from(tile.y), tile.z),
            action: tile.action,
            nbt: tile.nbt,
        })
    })
}

/// java17 `NHPC.handleSignEditorOpen` (`:1061-1075`): the world's tile entity at the position,
/// or a detached sign where there is none, handed to `displayGUIEditSign`, which opens the sign
/// editor for a sign and nothing for any other tile entity (java17
/// `client/entity/EntityPlayerSP.java:334-344`; its command-block branch opens the command-block
/// editor, which this client has no event for yet). java18 always opens the sign editor.
pub(super) fn sign_editor_open(
    state: &ClientState,
    open: &s2c5::SignEditorOpen,
    events: &mut Vec<ClientEvent>,
) {
    let position = BlockPos::new(open.x, open.y, open.z);
    // java17 `World.getTileEntity` -> `Chunk.func_150806_e`: the stored tile entity, or one the
    // block creates on demand; none for a block without one or an unloaded column.
    let opens_sign_editor = match state.world.tile_kind_at(position) {
        None => true,
        Some(TileKind::StandingSign | TileKind::WallSign) => true,
        // `BlockPistonMoving.createNewTileEntity` returns null: without a stored one there is
        // no tile entity, so the editor opens on a detached sign.
        Some(kind) if !kind.created_lazily() => state.world.tile_entity(position).is_none(),
        Some(_) => false,
    };
    if opens_sign_editor {
        events.push(ClientEvent::OpenSignEditor(position));
    }
}

/// java17 `NHPC.handleMaps` (`:1260-1265`): `ItemMap.loadMapData` creates `map_<id>` if needed,
/// then `MapData.updateMPMapData` applies the blob ([`s2c5::Maps::update`]): a colour column, the
/// icon list, or the scale, each alone. The client's map store takes java18's whole-packet form
/// (`S34PacketMaps.setMapdataTo`), so each 1.7.10 update is applied as a java18 packet carrying
/// the map's current scale and icons and, for colours, the one column the update writes.
///
/// A colour index is `(i + y) * 128 + x` with `x` up to 255, so a column at `x >= 128` is the one
/// at `x - 128` a row lower; past index 16383 1.7.10 throws after writing what fit, which is what
/// happens here too (the rows that fit, then an error).
pub(super) fn maps(
    state: &mut ClientState,
    maps: &s2c5::Maps,
    events: &mut Vec<ClientEvent>,
) -> Result<(), ClientError> {
    let (scale, icons) = match state.world.map_data(maps.map_id) {
        Some(data) => (
            data.scale,
            data.decorations
                .iter()
                .map(|decoration| s2c::MapIcon {
                    icon_type: decoration.icon_type,
                    x: decoration.x,
                    z: decoration.z,
                    direction: decoration.direction,
                })
                .collect(),
        ),
        None => (0, Vec::new()),
    };
    let mut packet = s2c::Maps {
        map_id: maps.map_id,
        scale,
        icons,
        update: None,
    };
    let mut overflow = None;
    let update = maps.update();
    match &update {
        Ok(MapUpdate::Colors { x, y_start, colors }) => {
            let column = usize::from(*x) % MAP_SIZE;
            let first_row = usize::from(*y_start) + usize::from(*x) / MAP_SIZE;
            let fits = MAP_SIZE.saturating_sub(first_row).min(colors.len());
            if fits < colors.len() {
                overflow = Some(colors.len() - fits);
            }
            if fits > 0 {
                packet.update = Some(s2c::MapPixelUpdate {
                    width: 1,
                    height: fits as u8,
                    min_x: column as u8,
                    min_y: first_row as u8,
                    data: colors[..fits].to_vec(),
                });
            }
        }
        // An icon's type is `(byte)(b >> 4)`; vanilla servers send 0-7, which both versions
        // draw alike (a negative type indexes past 1.7.10's icon sheet).
        Ok(MapUpdate::Icons(new_icons)) => {
            packet.icons = new_icons
                .iter()
                .map(|icon| s2c::MapIcon {
                    icon_type: (icon.kind as u8) & 15,
                    x: icon.x,
                    z: icon.z,
                    direction: icon.rotation,
                })
                .collect();
        }
        Ok(MapUpdate::Scale(new_scale)) => packet.scale = *new_scale,
        Ok(MapUpdate::Ignored) | Err(_) => {}
    }
    // The map record exists after this, as `loadMapData` creates it before the update runs.
    dispatch::apply(state, S2CPlay::Maps(packet), events)?;
    if let Err(error) = update {
        return Err(error.into());
    }
    if let Some(rows) = overflow {
        return Err(ClientError::Protocol(ProtoError::InvalidEnum {
            what: "map colour rows past the 128 x 128 image",
            value: rows as i32,
        }));
    }
    Ok(())
}
