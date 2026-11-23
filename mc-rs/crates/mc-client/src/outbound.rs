//! Outbound packet production: movement cadence, digging, placing, held-item sync, chat.
//!
//! java: `client/entity/EntityPlayerSP.onUpdateWalkingPlayer` (`:189`) and
//! `client/multiplayer/PlayerControllerMP.java`.
//!
//! # Hazard H11
//!
//! Movement cadence is anticheat-relevant *independently of whether the positions are correct*.
//! The decision of which of the four `C03PacketPlayer` shapes to send — including the `9.0E-4`
//! squared-distance threshold and the forced full update every twentieth tick — is a function
//! of physics state, and S13 already implements it exactly in
//! [`mc_physics::prediction::Predictor::outbound`]. **This module does not re-derive it.** It
//! translates the resulting [`Outbound`] into a packet and adds the two things
//! `onUpdateWalkingPlayer` does that are *not* physics: the sprint and sneak
//! `C0BPacketEntityAction` edges, which are sent only on a transition, before the movement
//! packet, and exactly once each.

use mc_math::{BlockPos, DVec3, Facing};
use mc_physics::prediction::Outbound;
use mc_protocol::play::{c2s, C2SPlay};
use mc_protocol::ItemStack;
use mc_ui::screens::sign::CommandBlockTarget;

/// Vanilla truncates a chat message to 100 characters in the packet's own constructor, before
/// it ever reaches the wire. java: `C01PacketChatMessage(String)`
pub const MAX_CHAT_LENGTH: usize = 100;

/// The cut with the 256 Character Chat mod on: 1.11's
/// serverbound chat limit, which multi-version servers accept. A server that really runs 1.8
/// rejects anything over [`MAX_CHAT_LENGTH`] (`readStringFromBuffer(100)`) and disconnects.
pub const LONG_CHAT_LENGTH: usize = 256;

/// The mutable bookkeeping `EntityPlayerSP` and `PlayerControllerMP` keep about what the server
/// has already been told.
///
/// Every field mirrors a specific vanilla field, and all of them exist to *suppress* packets:
/// sending a redundant sprint toggle or held-item change every tick is exactly the kind of
/// traffic anomaly hazard H11 warns about.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct OutboundState {
    /// java: `EntityPlayerSP.serverSprintState`
    server_sprinting: bool,
    /// java: `EntityPlayerSP.serverSneakState`
    server_sneaking: bool,
    /// java: `PlayerControllerMP.currentPlayerItem`, which has Java's default value `0`.
    server_held_slot: i32,
    /// The block of the last START this packet layer sent, cleared by STOP/ABORT and by a local
    /// instant/creative destroy. Approximates `PlayerControllerMP.currentBlock` while
    /// `isHittingBlock`; the live driver takes Java's exact ABORT target from the controller
    /// ([`Self::start_digging_after_abort`], [`Self::abort_digging_at`]).
    digging: Option<BlockPos>,
}

impl OutboundState {
    /// Fresh state for a new connection.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Resets only the report-edge fields owned by a newly constructed `EntityPlayerSP`.
    /// `PlayerControllerMP` survives respawn, so its held-slot and digging state stay intact.
    /// java: `Minecraft.setDimensionAndSpawnPlayer` / `EntityPlayerSP` field defaults.
    pub(crate) fn reset_local_player_reporting(&mut self) {
        self.server_sprinting = false;
        self.server_sneaking = false;
    }

    /// Resets only the fields owned by `PlayerControllerMP`, which `handleJoinGame` replaces
    /// on every JoinGame (`currentPlayerItem = 0`, no dig in progress). A mid-connection
    /// JoinGame keeps the `EntityPlayerSP`, so its two report-edge fields stay.
    pub(crate) fn reset_player_controller(&mut self) {
        self.server_held_slot = 0;
        self.digging = None;
    }

    /// The block currently being broken, if any. java: `PlayerControllerMP.getIsHittingBlock`
    #[must_use]
    pub fn digging(&self) -> Option<BlockPos> {
        self.digging
    }

    /// Everything `onUpdateWalkingPlayer` sends this tick.
    ///
    /// `movement` comes straight from [`mc_physics::prediction::Predictor::tick`]; `entity_id`
    /// is the local player's, which `C0BPacketEntityAction` carries. `watched_sprinting` is the
    /// local player's watched index-0 bit 3, as read by the session, rather than an independent
    /// outbound-only sprint flag.
    ///
    /// Order matters and is vanilla's: sprint edge, then sneak edge, then the movement packet.
    pub fn walking_update(
        &mut self,
        entity_id: i32,
        movement: Outbound,
        watched_sprinting: bool,
        sneaking: bool,
        out: &mut Vec<C2SPlay>,
    ) {
        self.walking_edges(entity_id, watched_sprinting, sneaking, out);
        out.push(movement_packet(movement));
    }

    /// Emit only the view-independent sprint/sneak edges from `onUpdateWalkingPlayer`.
    ///
    /// java: `EntityPlayerSP.onUpdateWalkingPlayer` sends these `C0B` transitions before its
    /// `isCurrentViewEntity()` guard.  A spectator camera target still has a locally simulated
    /// player and therefore still reports these edges, but must not report `C03`–`C06` for the
    /// target that is not the current view entity.
    pub fn walking_edges(
        &mut self,
        entity_id: i32,
        watched_sprinting: bool,
        sneaking: bool,
        out: &mut Vec<C2SPlay>,
    ) {
        if watched_sprinting != self.server_sprinting {
            out.push(entity_action(
                entity_id,
                if watched_sprinting {
                    c2s::EntityActionType::StartSprinting
                } else {
                    c2s::EntityActionType::StopSprinting
                },
            ));
            self.server_sprinting = watched_sprinting;
        }
        if sneaking != self.server_sneaking {
            out.push(entity_action(
                entity_id,
                if sneaking {
                    c2s::EntityActionType::StartSneaking
                } else {
                    c2s::EntityActionType::StopSneaking
                },
            ));
            self.server_sneaking = sneaking;
        }
    }

    /// Append the movement packet selected by `Predictor::outbound` without touching the C0B
    /// edge cache. Used by a caller that has already emitted the view-independent edges.
    pub(crate) fn movement_only(&self, movement: Outbound, out: &mut Vec<C2SPlay>) {
        out.push(movement_packet(movement));
    }

    // The riding branch (`EntityPlayerSP.onUpdate:174-179`, C05 + C0C) is built by
    // `Session::tick_riding`, which owns the state both packets read: the rider's own
    // `moveEntity` `onGround` and the action fields after the travel section's `*= 0.98F`. A
    // raw-`MoveInput` builder that used to live here sent neither and had no
    // caller, so it is gone rather than left as a second, non-vanilla way to encode a ride.

    /// Send `C09PacketHeldItemChange` if and only if the selected slot changed.
    /// java: `PlayerControllerMP.syncCurrentPlayItem` (`:379`), called at the top of
    /// `onPlayerDamageBlock` and `onPlayerRightClick`.
    pub fn sync_held_item(&mut self, slot: u8, out: &mut Vec<C2SPlay>) {
        if i32::from(slot) != self.server_held_slot {
            self.server_held_slot = i32::from(slot);
            out.push(C2SPlay::HeldItemChange(c2s::HeldItemChange {
                slot_id: i16::from(slot),
            }));
        }
    }

    /// Begin breaking a block.
    ///
    /// java: `PlayerControllerMP.clickBlock` (`:198`). Switching target while already digging
    /// sends an `ABORT_DESTROY_BLOCK` for the old block first — leaving it out makes the server
    /// keep accumulating damage on a block the client has stopped looking at.
    pub fn start_digging(&mut self, position: BlockPos, face: Facing, out: &mut Vec<C2SPlay>) {
        if let Some(previous) = self.digging {
            out.push(digging(
                c2s::PlayerDiggingStatus::AbortDestroyBlock,
                previous,
                face,
            ));
        }
        out.push(digging(
            c2s::PlayerDiggingStatus::StartDestroyBlock,
            position,
            face,
        ));
        self.digging = Some(position);
    }

    /// [`Self::start_digging`] with the ABORT decided by the controller rather than by this
    /// packet layer's own record of the last START.
    ///
    /// java: `clickBlock` (`:236-243`) sends `ABORT_DESTROY_BLOCK(this.currentBlock, face)`
    /// exactly when `isHittingBlock`, then the START. Only `PlayerControllerMP` knows those two
    /// fields: after an instant break interrupted a dig, `isHittingBlock` is still true and
    /// `currentBlock` is `onPlayerDestroyBlock`'s `(x, -1, z)`, and the creative branch never
    /// ABORTs at all. [`crate::app::interact::PlayerController::take_pending_abort`] carries the
    /// controller's answer (H3).
    pub fn start_digging_after_abort(
        &mut self,
        abort: Option<BlockPos>,
        position: BlockPos,
        face: Facing,
        out: &mut Vec<C2SPlay>,
    ) {
        if let Some(previous) = abort {
            out.push(digging(
                c2s::PlayerDiggingStatus::AbortDestroyBlock,
                previous,
                face,
            ));
        }
        out.push(digging(
            c2s::PlayerDiggingStatus::StartDestroyBlock,
            position,
            face,
        ));
        self.digging = Some(position);
    }

    /// [`Self::abort_digging`] for the controller's `currentBlock`. java: `resetBlockRemoving`
    /// (`:278`) names `this.currentBlock`, which may be a `y = -1` position, with `DOWN`.
    pub fn abort_digging_at(&mut self, position: BlockPos, out: &mut Vec<C2SPlay>) {
        out.push(digging(
            c2s::PlayerDiggingStatus::AbortDestroyBlock,
            position,
            Facing::Down,
        ));
        self.digging = None;
    }

    /// Forget an active dig after a local instant/creative destroy that follows START without a
    /// STOP. Otherwise the next unrelated START would emit a spurious ABORT.
    pub fn clear_digging(&mut self) {
        self.digging = None;
    }

    /// Report that the block finished breaking.
    /// java: the `curBlockDamageMP >= 1.0F` branch of `onPlayerDamageBlock` (`:321`).
    pub fn finish_digging(&mut self, position: BlockPos, face: Facing, out: &mut Vec<C2SPlay>) {
        out.push(digging(
            c2s::PlayerDiggingStatus::StopDestroyBlock,
            position,
            face,
        ));
        self.digging = None;
    }

    /// Stop breaking without finishing. A no-op if nothing was being broken, matching the
    /// `if (this.isHittingBlock)` guard. java: `resetBlockRemoving` (`:274`), which always uses
    /// `EnumFacing.DOWN`.
    pub fn abort_digging(&mut self, out: &mut Vec<C2SPlay>) {
        if let Some(position) = self.digging.take() {
            out.push(digging(
                c2s::PlayerDiggingStatus::AbortDestroyBlock,
                position,
                Facing::Down,
            ));
        }
    }

    /// Right-click a block.
    ///
    /// `cursor` is the hit point **relative to the block's corner**, `0.0..=1.0` on each axis.
    /// java: `onPlayerRightClick` (`:390`) computes it as `hitVec - hitPos` and
    /// `C08PacketPlayerBlockPlacement.writePacketData` sends `(int)(f * 16.0F)` per axis.
    pub fn place_block(
        &mut self,
        position: BlockPos,
        face: Facing,
        cursor: DVec3,
        held: Option<ItemStack>,
        held_slot: u8,
        out: &mut Vec<C2SPlay>,
    ) {
        self.sync_held_item(held_slot, out);
        out.push(C2SPlay::PlayerBlockPlacement(c2s::PlayerBlockPlacement {
            position,
            placed_block_direction: face.index(),
            stack: held,
            cursor_x: quantise_cursor(cursor.x),
            cursor_y: quantise_cursor(cursor.y),
            cursor_z: quantise_cursor(cursor.z),
        }));
    }

    /// Use the held item with no block target (eat, draw a bow, throw a pearl).
    ///
    /// java: `sendUseItem` (`:456`) is `syncCurrentPlayItem()` *then*
    /// `C08PacketPlayerBlockPlacement(stack)`, whose no-target constructor uses position
    /// `(-1, -1, -1)` and direction `255`. The held-item sync was missing here — one of the
    /// seven non-hotbar `syncCurrentPlayItem` call sites vanilla has. It is
    /// suppressed when the server already knows the slot, so on the common path it adds nothing
    /// to the wire.
    pub fn use_item(&mut self, held_slot: u8, held: Option<ItemStack>, out: &mut Vec<C2SPlay>) {
        self.sync_held_item(held_slot, out);
        out.push(C2SPlay::PlayerBlockPlacement(c2s::PlayerBlockPlacement {
            position: BlockPos::new(-1, -1, -1),
            placed_block_direction: 255,
            stack: held,
            cursor_x: 0,
            cursor_y: 0,
            cursor_z: 0,
        }));
    }

    /// Stop using a chargeable item — the packet that actually *fires* a drawn bow, swallows the
    /// food and drinks the potion.
    ///
    /// java: `onStoppedUsingItem` (`:576`):
    ///
    /// ```text
    /// this.syncCurrentPlayItem();
    /// this.netClientHandler.addToSendQueue(new C07PacketPlayerDigging(RELEASE_USE_ITEM,
    ///                                          BlockPos.ORIGIN, EnumFacing.DOWN));
    /// playerIn.stopUsingItem();
    /// ```
    ///
    /// The `BlockPos.ORIGIN`/`DOWN` payload is ignored by the server for this status, but it is
    /// what vanilla puts on the wire and an anticheat can read it (hazard H11). The
    /// `syncCurrentPlayItem` that precedes it is vanilla's, and matters: a server that thinks
    /// the player is holding a different slot resolves the release against the wrong item.
    ///
    /// `stopUsingItem`'s local half is the caller's — [`crate::app::interact::PlayerController`]
    /// owns `itemInUse`.
    pub fn release_use_item(&mut self, held_slot: u8, out: &mut Vec<C2SPlay>) {
        self.sync_held_item(held_slot, out);
        out.push(digging(
            c2s::PlayerDiggingStatus::ReleaseUseItem,
            BlockPos::new(0, 0, 0),
            Facing::Down,
        ));
    }

    /// Drop the held stack, or one item from it.
    /// java: `EntityPlayerSP.dropOneItem` → `C07PacketPlayerDigging` with `DROP_ITEM` /
    /// `DROP_ALL_ITEMS`, `BlockPos.ORIGIN`, `EnumFacing.DOWN`.
    pub fn drop_item(&mut self, whole_stack: bool, out: &mut Vec<C2SPlay>) {
        out.push(digging(
            if whole_stack {
                c2s::PlayerDiggingStatus::DropAllItems
            } else {
                c2s::PlayerDiggingStatus::DropItem
            },
            BlockPos::new(0, 0, 0),
            Facing::Down,
        ));
    }

    /// Swing the arm. java: `EntityPlayerSP.swingItem` → `C0APacketAnimation`
    pub fn swing_arm(&mut self, out: &mut Vec<C2SPlay>) {
        out.push(C2SPlay::Animation(c2s::Animation));
    }

    /// Attack an entity. java: `PlayerControllerMP.attackEntity` (`:495`) — the swing is a
    /// separate packet the caller also sends.
    ///
    /// ```text
    /// this.syncCurrentPlayItem();
    /// this.netClientHandler.addToSendQueue(new C02PacketUseEntity(targetEntity, Action.ATTACK));
    /// ```
    ///
    /// The `syncCurrentPlayItem()` is vanilla's and was missing: it is one of the seven
    /// non-hotbar `C09PacketHeldItemChange` call sites, and a server that has not been told the
    /// slot resolves the hit with the wrong item's damage.
    pub fn attack_entity(&mut self, entity_id: i32, slot: u8, out: &mut Vec<C2SPlay>) {
        self.sync_held_item(slot, out);
        out.push(C2SPlay::UseEntity(c2s::UseEntity {
            entity_id,
            action: c2s::UseEntityAction::Attack,
        }));
    }

    /// Right-click an entity: `INTERACT_AT`, followed by `INTERACT` unless the local
    /// `EntityArmorStand.interactAt` implementation consumed the click.
    ///
    /// java: `Minecraft.rightClickMouse`'s `case ENTITY` (`:1583-1592`) calls two
    /// `PlayerControllerMP` methods in an `if`/`else if`, and each one sends a packet before it
    /// returns anything:
    ///
    /// ```text
    /// public boolean isPlayerRightClickingOnEntity(player, entityIn, movingObject) {
    ///     this.syncCurrentPlayItem();
    ///     Vec3 vec3 = new Vec3(movingObject.hitVec.xCoord - entityIn.posX, ...);
    ///     this.netClientHandler.addToSendQueue(new C02PacketUseEntity(entityIn, vec3));
    ///     return this.currentGameType != SPECTATOR && entityIn.interactAt(player, vec3);
    /// }
    /// public boolean interactWithEntitySendPacket(playerIn, targetEntity) {
    ///     this.syncCurrentPlayItem();
    ///     this.netClientHandler.addToSendQueue(new C02PacketUseEntity(targetEntity, Action.INTERACT));
    ///     return this.currentGameType != SPECTATOR && playerIn.interactWith(targetEntity);
    /// }
    /// ```
    ///
    /// So `INTERACT_AT` goes first, carrying the click point **relative to the entity's
    /// position**, and `INTERACT` follows unless `interactAt` returned true. Only
    /// `EntityArmorStand` overrides `interactAt` in 1.8.9 (`Entity.interactAt` is
    /// `return false`). The caller supplies that one client-side result because it owns the
    /// tracked entity type and watched marker flag. W-USEENTITY:
    /// `F-P0-client.multiplayer-01-1`, `F-P0-network.play-02-7`, `F-P0-network.play.client-01-1`,
    /// `F-P1-client-01-4`, `F-P1-entity.item-02-2`, `F-P1-entity.item-06-3`,
    /// `F-P1-entity.player-01-2`, `F-P1-entity.player-01-9`, `F-P1-item-11-8`,
    /// `F-P1-util-12-2`.
    ///
    /// The second `sync_held_item` is not redundant bookkeeping: it is vanilla's own second
    /// call, and it emits nothing because the slot cannot have changed between the two.
    pub fn use_entity_right_click(
        &mut self,
        entity_id: i32,
        slot: u8,
        target: [f32; 3],
        send_interact: bool,
        out: &mut Vec<C2SPlay>,
    ) {
        self.sync_held_item(slot, out);
        out.push(C2SPlay::UseEntity(c2s::UseEntity {
            entity_id,
            action: c2s::UseEntityAction::InteractAt {
                target_x: target[0],
                target_y: target[1],
                target_z: target[2],
            },
        }));
        if send_interact {
            self.sync_held_item(slot, out);
            out.push(C2SPlay::UseEntity(c2s::UseEntity {
                entity_id,
                action: c2s::UseEntityAction::Interact,
            }));
        }
    }
}

/// Build the chat packet, applying vanilla's 100-character truncation.
///
/// java: `C01PacketChatMessage(String)` truncates with `substring(0, 100)`, which counts **UTF-16
/// code units**. Truncating by Rust `char`s instead would let a message of 100 astral characters
/// through where vanilla would have cut it at 50, so the boundary is computed in UTF-16 units
/// and then mapped back to a `char` boundary.
#[must_use]
pub fn chat_message(message: &str) -> C2SPlay {
    chat_message_with_limit(message, MAX_CHAT_LENGTH)
}

/// [`chat_message`] with the cut at `limit` UTF-16 units instead of 100: the 256 Character Chat
/// mod's [`LONG_CHAT_LENGTH`] for chat the player typed.
#[must_use]
pub fn chat_message_with_limit(message: &str, limit: usize) -> C2SPlay {
    C2SPlay::ChatMessage(c2s::ChatMessage {
        message: truncate_utf16(message, limit),
    })
}

/// Reproduce Java's `substring(0, limit)` followed by UTF-8 encoding.
///
/// Rust strings cannot contain Java's lone UTF-16 surrogates. When the exact Java boundary
/// splits an astral code point, its UTF-8 encoder replaces the orphaned high surrogate with
/// `?`, so this returns that wire-equivalent scalar instead.
fn truncate_utf16(text: &str, limit: usize) -> String {
    let mut units = 0usize;
    for (offset, ch) in text.char_indices() {
        let width = ch.len_utf16();
        if units + width > limit {
            let mut prefix = text[..offset].to_owned();
            if units < limit {
                prefix.push('?');
            }
            return prefix;
        }
        units += width;
    }
    text.to_owned()
}

/// The `C15PacketClientSettings` vanilla sends immediately after `JoinGame`.
/// java: `GameSettings.sendSettingsToServer` (`:1203`)
#[must_use]
pub fn client_settings(
    locale: &str,
    view_distance: i8,
    chat_visibility: c2s::ChatVisibility,
    chat_colors: bool,
    displayed_skin_parts: u8,
) -> C2SPlay {
    C2SPlay::ClientSettings(c2s::ClientSettings {
        locale: locale.to_owned(),
        view_distance,
        chat_visibility,
        chat_colors,
        displayed_skin_parts,
    })
}

/// The `MC|Brand` plugin message vanilla sends from `handleJoinGame` (`:291`).
///
/// The payload is a protocol string (VarInt length prefix), not raw bytes — vanilla writes it
/// with `new PacketBuffer(Unpooled.buffer()).writeString(...)`.
#[must_use]
pub fn brand_message(brand: &str) -> C2SPlay {
    let mut payload = mc_protocol::PacketBuffer::new();
    payload.write_string(brand);
    C2SPlay::PluginMessage(c2s::PluginMessage {
        channel: "MC|Brand".to_owned(),
        data: payload.as_slice().to_vec(),
    })
}

/// Send the current anvil rename field. java: `GuiRepair.renameItem` -> `MC|ItemName`.
#[must_use]
pub fn anvil_name(name: &str) -> C2SPlay {
    let mut payload = mc_protocol::PacketBuffer::new();
    payload.write_string(name);
    C2SPlay::PluginMessage(c2s::PluginMessage {
        channel: "MC|ItemName".to_owned(),
        data: payload.as_slice().to_vec(),
    })
}

/// Send the complete edited/signed book stack in the vanilla custom-payload body.
/// java: `GuiScreenBook.sendBookToServer` -> `C17PacketCustomPayload`.
#[must_use]
pub fn book_payload(channel: &str, stack: &mc_protocol::ItemStack) -> C2SPlay {
    let mut payload = mc_protocol::PacketBuffer::new();
    mc_protocol::ItemStack::write(Some(stack), &mut payload);
    C2SPlay::PluginMessage(c2s::PluginMessage {
        channel: channel.to_owned(),
        data: payload.as_slice().to_vec(),
    })
}

/// Send an edited command block or command minecart over `MC|AdvCdm`.
///
/// java: `GuiCommandBlock.actionPerformed` (`:79-110`),
/// `TileEntityCommandBlock.func_145757_a` (`:43-48`) and
/// `EntityMinecartCommandBlock.func_145757_a` (`:28-31`).  Mode `0` writes three separate
/// big-endian `i32` coordinates; mode `1` writes one big-endian entity id.  Neither target uses
/// the packed protocol `Position` long.  The command then uses the ordinary protocol-string
/// VarInt length prefix, followed by one boolean byte for `TrackOutput`.
#[must_use]
pub fn command_block_payload(
    target: CommandBlockTarget,
    command: &str,
    track_output: bool,
) -> C2SPlay {
    let mut payload = mc_protocol::PacketBuffer::new();
    match target {
        CommandBlockTarget::Block { position } => {
            payload.write_u8(0);
            payload.write_i32(position.x);
            payload.write_i32(position.y);
            payload.write_i32(position.z);
        }
        CommandBlockTarget::Minecart { entity_id } => {
            payload.write_u8(1);
            payload.write_i32(entity_id);
        }
    }
    payload.write_string(command);
    payload.write_bool(track_output);
    C2SPlay::PluginMessage(c2s::PluginMessage {
        channel: "MC|AdvCdm".to_owned(),
        data: payload.as_slice().to_vec(),
    })
}

/// Tell the server the client toggled flight. java: `EntityPlayerSP.sendPlayerAbilities`
/// (`client/entity/EntityPlayerSP.java:394`) → `C13PacketPlayerAbilities(this.capabilities)`.
///
/// Sent when spectator mode forces flight on (`onLivingUpdate:829`), on both edges
/// of the double-tap-jump toggle (`:841`), and when landing clears flight (`:906`). The packet
/// carries the whole capability set, although vanilla's server consumes only the flying flag
/// (and gates it with its own `allowFlying`) in `NetHandlerPlayServer:1206-1209`.
#[must_use]
pub fn player_abilities(abilities: &crate::sync::Abilities) -> C2SPlay {
    C2SPlay::PlayerAbilities(c2s::PlayerAbilities {
        invulnerable: abilities.invulnerable,
        flying: abilities.flying,
        allow_flying: abilities.allow_flying,
        creative_mode: abilities.creative_mode,
        fly_speed: abilities.fly_speed,
        walk_speed: abilities.walk_speed,
    })
}

/// Acknowledge one stage of a server-supplied resource pack.
///
/// java: `new C19PacketResourcePackStatus(hash, action)`. The protocol writer owns Java's
/// 40-UTF-16-unit constructor limit, so callers can pass the hash exactly as S48 supplied it.
/// Keeping this constructor here makes the two-packet accept path visibly ordered at its live
/// driver call site instead of hand-building subtly different packets in each policy branch.
#[must_use]
pub fn resource_pack_status(
    hash: impl Into<String>,
    result: c2s::ResourcePackStatusResult,
) -> C2SPlay {
    C2SPlay::ResourcePackStatus(c2s::ResourcePackStatus {
        hash: hash.into(),
        result,
    })
}

/// Ask to respawn after death. java: `EntityPlayerSP.respawnPlayer` →
/// `C16PacketClientStatus(PERFORM_RESPAWN)`
#[must_use]
pub fn perform_respawn() -> C2SPlay {
    C2SPlay::ClientStatus(c2s::ClientStatus {
        action: c2s::ClientStatusAction::PerformRespawn,
    })
}

/// Translate S13's cadence decision into the packet it names.
/// java: the four branches of `onUpdateWalkingPlayer` (`:232-247`).
#[must_use]
pub fn movement_packet(movement: Outbound) -> C2SPlay {
    match movement {
        Outbound::Ground { on_ground } => C2SPlay::Player(c2s::Player { on_ground }),
        Outbound::Position { pos, on_ground } => C2SPlay::PlayerPosition(c2s::PlayerPosition {
            x: pos.x,
            y: pos.y,
            z: pos.z,
            on_ground,
        }),
        Outbound::Look {
            yaw,
            pitch,
            on_ground,
        } => C2SPlay::PlayerLook(c2s::PlayerLook {
            yaw,
            pitch,
            on_ground,
        }),
        Outbound::PositionLook {
            pos,
            yaw,
            pitch,
            on_ground,
        } => C2SPlay::PlayerPositionAndLook(c2s::PlayerPositionAndLook {
            x: pos.x,
            y: pos.y,
            z: pos.z,
            yaw,
            pitch,
            on_ground,
        }),
    }
}

/// The `C06PacketPlayerPosLook` a client must send back to confirm a server correction.
/// java: `handlePlayerPosLook` (`:717`), which replies with `onGround = false` unconditionally.
#[must_use]
pub fn confirm_position(pos: DVec3, yaw: f32, pitch: f32) -> C2SPlay {
    C2SPlay::PlayerPositionAndLook(c2s::PlayerPositionAndLook {
        x: pos.x,
        y: pos.y,
        z: pos.z,
        yaw,
        pitch,
        on_ground: false,
    })
}

/// Tell the server that a window which could not be represented locally is no longer open.
///
/// java: `EntityPlayerSP.closeScreen` (`:330-333`) queues
/// `new C0DPacketCloseWindow(this.openContainer.windowId)` before tearing down the local GUI.
/// The open-window host uses the packet's id directly for an unsupported `S2D` branch, because
/// there is no local `Container` from which to read `openContainer.windowId`.
#[must_use]
pub fn close_window(window_id: u8) -> C2SPlay {
    C2SPlay::CloseWindow(c2s::CloseWindow { window_id })
}

/// `C0CPacketInput`, whose two booleans are bit `0` and bit `1` of one byte.
/// java: `C0CPacketInput.writePacketData` (`:44-60`).
#[must_use]
pub fn steer_vehicle(sideways: f32, forward: f32, jump: bool, sneak: bool) -> C2SPlay {
    C2SPlay::SteerVehicle(c2s::SteerVehicle {
        sideways,
        forward,
        flags: u8::from(jump) | (u8::from(sneak) << 1),
    })
}

/// Ask the server to open the ridden horse's inventory.
/// java: `EntityPlayerSP.sendHorseInventory` (`:412`), reached from `Minecraft.java:2094`'s
/// `if (this.playerController.isRidingHorse())` fork on the inventory key.
#[must_use]
pub fn horse_inventory(entity_id: i32) -> C2SPlay {
    entity_action(entity_id, c2s::EntityActionType::OpenInventory)
}

/// Release a charged horse jump.
///
/// java: `EntityPlayerSP.sendHorseJump` (`:407`) —
/// `new C0BPacketEntityAction(this, RIDING_JUMP, (int)(this.getHorseJumpPower() * 100.0F))`.
/// The cast is a **truncation toward zero**, not a round. The charge curve
/// (`onLivingUpdate:872-893`) is `counter * 0.1F` for the first ten ticks and then
/// `0.8F + 2.0F / (counter - 9) * 0.1F`, which **peaks at `1.0` on tick 10 and decays back
/// toward `0.8`** — so holding jump longer past that point is worse, not better, and `100` is
/// reachable in exactly one tick.
#[must_use]
pub fn horse_jump(entity_id: i32, power: f32) -> C2SPlay {
    C2SPlay::EntityAction(c2s::EntityAction {
        entity_id,
        action: c2s::EntityActionType::RidingJump,
        aux_data: (power * 100.0) as i32,
    })
}

fn entity_action(entity_id: i32, action: c2s::EntityActionType) -> C2SPlay {
    C2SPlay::EntityAction(c2s::EntityAction {
        entity_id,
        action,
        aux_data: 0,
    })
}

fn digging(status: c2s::PlayerDiggingStatus, position: BlockPos, face: Facing) -> C2SPlay {
    C2SPlay::PlayerDigging(c2s::PlayerDigging {
        status,
        position,
        face,
    })
}

/// `(int)(f * 16.0F)` then `writeByte`: the hit offset is first narrowed from `double` to
/// `float` in `PlayerControllerMP.onPlayerRightClick`, then scaled, truncated toward zero and
/// narrowed to eight bits.
fn quantise_cursor(value: f64) -> u8 {
    ((value as f32 * 16.0) as i32) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    fn drain(f: impl FnOnce(&mut OutboundState, &mut Vec<C2SPlay>)) -> Vec<C2SPlay> {
        let mut state = OutboundState::new();
        let mut out = Vec::new();
        f(&mut state, &mut out);
        out
    }

    /// The four shapes, each mapped to the packet `onUpdateWalkingPlayer` names for it.
    #[test]
    fn every_outbound_shape_maps_to_its_vanilla_packet() {
        assert!(matches!(
            movement_packet(Outbound::Ground { on_ground: true }),
            C2SPlay::Player(c2s::Player { on_ground: true })
        ));
        assert!(matches!(
            movement_packet(Outbound::Position {
                pos: DVec3::new(1.0, 2.0, 3.0),
                on_ground: false
            }),
            C2SPlay::PlayerPosition(_)
        ));
        assert!(matches!(
            movement_packet(Outbound::Look {
                yaw: 1.0,
                pitch: 2.0,
                on_ground: true
            }),
            C2SPlay::PlayerLook(_)
        ));
        assert!(matches!(
            movement_packet(Outbound::PositionLook {
                pos: DVec3::ZERO,
                yaw: 0.0,
                pitch: 0.0,
                on_ground: true
            }),
            C2SPlay::PlayerPositionAndLook(_)
        ));
    }

    /// Y on the wire is the bounding box's `minY`, i.e. the feet — which is exactly what
    /// `PhysicsState::pos` and therefore `Outbound::Position.pos` already carry.
    #[test]
    fn position_packets_carry_the_predictors_coordinates_unchanged() {
        let pos = DVec3::new(-12.5, 63.0, 1024.125);
        match movement_packet(Outbound::Position {
            pos,
            on_ground: true,
        }) {
            C2SPlay::PlayerPosition(packet) => {
                assert_eq!((packet.x, packet.y, packet.z), (pos.x, pos.y, pos.z));
                assert!(packet.on_ground);
            }
            other => panic!("{other:?}"),
        }
    }

    /// Hazard H11's call-site half: the sprint/sneak actions fire on the **edge** only, and the
    /// movement packet always follows them.
    #[test]
    fn sprint_and_sneak_actions_fire_only_on_a_transition() {
        let mut state = OutboundState::new();
        let mut out = Vec::new();

        state.walking_update(
            7,
            Outbound::Ground { on_ground: true },
            false,
            false,
            &mut out,
        );
        assert_eq!(out.len(), 1, "no edges on the first tick");
        out.clear();

        state.walking_update(
            7,
            Outbound::Ground { on_ground: true },
            true,
            false,
            &mut out,
        );
        assert_eq!(out.len(), 2);
        assert!(matches!(
            out[0],
            C2SPlay::EntityAction(c2s::EntityAction {
                entity_id: 7,
                action: c2s::EntityActionType::StartSprinting,
                aux_data: 0
            })
        ));
        assert!(matches!(out[1], C2SPlay::Player(_)));
        out.clear();

        // Still sprinting: no second StartSprinting.
        for _ in 0..5 {
            state.walking_update(
                7,
                Outbound::Ground { on_ground: true },
                true,
                false,
                &mut out,
            );
        }
        assert_eq!(out.len(), 5);
        assert!(out.iter().all(|p| matches!(p, C2SPlay::Player(_))));
        out.clear();

        state.walking_update(
            7,
            Outbound::Ground { on_ground: true },
            false,
            true,
            &mut out,
        );
        assert_eq!(out.len(), 3);
        assert!(matches!(
            out[0],
            C2SPlay::EntityAction(c2s::EntityAction {
                action: c2s::EntityActionType::StopSprinting,
                ..
            })
        ));
        assert!(matches!(
            out[1],
            C2SPlay::EntityAction(c2s::EntityAction {
                action: c2s::EntityActionType::StartSneaking,
                ..
            })
        ));
    }

    #[test]
    fn foreign_view_edges_do_not_require_a_movement_packet() {
        let mut state = OutboundState::new();
        let mut out = Vec::new();
        state.walking_edges(42, true, true, &mut out);
        assert!(matches!(
            out.as_slice(),
            [
                C2SPlay::EntityAction(c2s::EntityAction {
                    action: c2s::EntityActionType::StartSprinting,
                    ..
                }),
                C2SPlay::EntityAction(c2s::EntityAction {
                    action: c2s::EntityActionType::StartSneaking,
                    ..
                })
            ]
        ));
        assert!(!out.iter().any(|packet| matches!(
            packet,
            C2SPlay::Player(_)
                | C2SPlay::PlayerPosition(_)
                | C2SPlay::PlayerLook(_)
                | C2SPlay::PlayerPositionAndLook(_)
        )));
    }

    #[test]
    fn respawn_resets_player_report_edges_but_preserves_controller_bookkeeping() {
        let mut state = OutboundState::new();
        let target = BlockPos::new(4, 65, 7);
        let mut out = Vec::new();
        state.walking_update(
            7,
            Outbound::Ground { on_ground: true },
            true,
            true,
            &mut out,
        );
        state.sync_held_item(4, &mut out);
        state.start_digging(target, Facing::Up, &mut out);
        out.clear();

        state.reset_local_player_reporting();
        assert_eq!(state.server_held_slot, 4);
        assert_eq!(state.digging(), Some(target));
        state.walking_update(
            7,
            Outbound::Ground { on_ground: true },
            true,
            true,
            &mut out,
        );
        assert!(matches!(
            out[0],
            C2SPlay::EntityAction(c2s::EntityAction {
                action: c2s::EntityActionType::StartSprinting,
                ..
            })
        ));
        assert!(matches!(
            out[1],
            C2SPlay::EntityAction(c2s::EntityAction {
                action: c2s::EntityActionType::StartSneaking,
                ..
            })
        ));
        assert!(matches!(out[2], C2SPlay::Player(_)));
        state.sync_held_item(4, &mut out);
        assert_eq!(out.len(), 3, "controller's held slot survives respawn");
    }

    /// java: `currentPlayerItem` has no initializer, so Java defaults it to slot 0.
    #[test]
    fn held_item_sync_starts_at_vanillas_zero_and_fires_once_per_change() {
        let mut state = OutboundState::new();
        let mut out = Vec::new();

        state.sync_held_item(0, &mut out);
        assert!(out.is_empty(), "fresh vanilla state already knows slot 0");
        state.sync_held_item(3, &mut out);
        assert_eq!(out.len(), 1);
        assert!(matches!(
            out[0],
            C2SPlay::HeldItemChange(c2s::HeldItemChange { slot_id: 3 })
        ));
        state.sync_held_item(3, &mut out);
        assert_eq!(out.len(), 1, "no packet when nothing changed");
        state.sync_held_item(0, &mut out);
        assert_eq!(out.len(), 2, "returning to slot 0 is a real edge");
    }

    /// java: `clickBlock` aborts the previous target before starting a new one.
    #[test]
    fn switching_dig_target_aborts_the_previous_one_first() {
        let mut state = OutboundState::new();
        let mut out = Vec::new();
        let a = BlockPos::new(1, 2, 3);
        let b = BlockPos::new(4, 5, 6);

        state.start_digging(a, Facing::Up, &mut out);
        assert_eq!(out.len(), 1);
        assert_eq!(state.digging(), Some(a));

        // The controller calls us only after it has decided the dig identity changed. It may
        // therefore request a restart at the same position, which is ABORT then START.
        state.start_digging(a, Facing::East, &mut out);
        assert_eq!(out.len(), 3);
        for (packet, status) in [
            (&out[1], c2s::PlayerDiggingStatus::AbortDestroyBlock),
            (&out[2], c2s::PlayerDiggingStatus::StartDestroyBlock),
        ] {
            let C2SPlay::PlayerDigging(packet) = packet else {
                panic!("expected digging packet, got {packet:?}");
            };
            assert_eq!(packet.status, status);
            assert_eq!(packet.position, a);
            assert_eq!(
                packet.face,
                Facing::East,
                "new click face belongs to both C07s"
            );
        }

        state.start_digging(b, Facing::North, &mut out);
        assert_eq!(out.len(), 5);
        match &out[3] {
            C2SPlay::PlayerDigging(packet) => {
                assert_eq!(packet.status, c2s::PlayerDiggingStatus::AbortDestroyBlock);
                assert_eq!(packet.position, a);
                assert_eq!(packet.face, Facing::North);
            }
            other => panic!("{other:?}"),
        }
        match &out[4] {
            C2SPlay::PlayerDigging(packet) => {
                assert_eq!(packet.status, c2s::PlayerDiggingStatus::StartDestroyBlock);
                assert_eq!(packet.position, b);
            }
            other => panic!("{other:?}"),
        }
    }

    /// java: `resetBlockRemoving`'s `if (this.isHittingBlock)` guard, and its hard-coded DOWN.
    #[test]
    fn abort_is_a_no_op_when_not_digging_and_always_faces_down() {
        let mut state = OutboundState::new();
        let mut out = Vec::new();
        state.abort_digging(&mut out);
        assert!(out.is_empty());

        state.start_digging(BlockPos::new(0, 1, 0), Facing::East, &mut out);
        out.clear();
        state.abort_digging(&mut out);
        assert_eq!(out.len(), 1);
        match &out[0] {
            C2SPlay::PlayerDigging(packet) => assert_eq!(packet.face, Facing::Down),
            other => panic!("{other:?}"),
        }
        assert_eq!(state.digging(), None);
        // And a second abort is silent again.
        out.clear();
        state.abort_digging(&mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn finish_digging_clears_the_target() {
        let mut state = OutboundState::new();
        let mut out = Vec::new();
        let pos = BlockPos::new(2, 3, 4);
        state.start_digging(pos, Facing::Up, &mut out);
        state.finish_digging(pos, Facing::Up, &mut out);
        assert_eq!(state.digging(), None);
        match out.last().unwrap() {
            C2SPlay::PlayerDigging(packet) => {
                assert_eq!(packet.status, c2s::PlayerDiggingStatus::StopDestroyBlock);
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn local_instant_destroy_clears_dig_state_before_the_next_start() {
        let mut state = OutboundState::new();
        let mut out = Vec::new();
        let instant = BlockPos::new(2, 3, 4);
        let next = BlockPos::new(5, 6, 7);

        state.start_digging(instant, Facing::Up, &mut out);
        state.clear_digging();
        state.start_digging(next, Facing::North, &mut out);

        assert_eq!(state.digging(), Some(next));
        assert_eq!(
            out.len(),
            2,
            "the next START must not inherit an old local destroy"
        );
        assert!(matches!(
            out[1],
            C2SPlay::PlayerDigging(c2s::PlayerDigging {
                status: c2s::PlayerDiggingStatus::StartDestroyBlock,
                position,
                face: Facing::North,
            }) if position == next
        ));
    }

    /// java: `PlayerControllerMP.onPlayerRightClick` first casts the `double` hit offset to
    /// `float`; `C08.writePacketData` then multiplies that float by `16.0F` and truncates it.
    #[test]
    fn cursor_coordinates_narrow_to_float_before_sixteenth_truncation() {
        assert_eq!(quantise_cursor(0.0), 0);
        assert_eq!(quantise_cursor(0.5), 8);
        assert_eq!(quantise_cursor(1.0), 16);
        assert_eq!(quantise_cursor(0.999), 15, "0.999 * 16 = 15.984 -> 15");
        assert_eq!(quantise_cursor(0.0624), 0, "just under 1/16");

        // Negative controls for the f64 mutant: these doubles are below a sixteenth boundary,
        // but Java's explicit float cast rounds them up before the multiply.
        assert_eq!(quantise_cursor(0.999_999_999_9), 16);
        assert_eq!(quantise_cursor(0.062_499_999_99), 1);

        // Nearby representable floats remain below their boundaries, proving this is not a
        // blanket ceil/round operation.
        assert_eq!(quantise_cursor(0.999_999_94), 15);
        assert_eq!(quantise_cursor(0.062_499_996), 0);
    }

    #[test]
    fn place_block_syncs_the_slot_and_encodes_the_hit_point() {
        let out = drain(|state, out| {
            state.place_block(
                BlockPos::new(10, 64, -3),
                Facing::Up,
                DVec3::new(0.5, 1.0, 0.25),
                None,
                2,
                out,
            );
        });
        assert_eq!(out.len(), 2, "held-item sync then placement");
        assert!(matches!(
            out[0],
            C2SPlay::HeldItemChange(c2s::HeldItemChange { slot_id: 2 })
        ));
        match &out[1] {
            C2SPlay::PlayerBlockPlacement(packet) => {
                assert_eq!(packet.position, BlockPos::new(10, 64, -3));
                assert_eq!(packet.placed_block_direction, 1, "EnumFacing.UP.getIndex()");
                assert_eq!(
                    (packet.cursor_x, packet.cursor_y, packet.cursor_z),
                    (8, 16, 4)
                );
            }
            other => panic!("{other:?}"),
        }
    }

    /// java: `EnumFacing`'s ordinal order, which `getIndex()` returns verbatim.
    #[test]
    fn facing_indices_are_the_vanilla_ordinals() {
        assert_eq!(Facing::Down.index(), 0);
        assert_eq!(Facing::Up.index(), 1);
        assert_eq!(Facing::North.index(), 2);
        assert_eq!(Facing::South.index(), 3);
        assert_eq!(Facing::West.index(), 4);
        assert_eq!(Facing::East.index(), 5);
    }

    /// java: `sendUseItem`'s no-target constructor, and the `syncCurrentPlayItem()` that runs
    /// before it.
    #[test]
    fn use_item_syncs_the_slot_then_targets_the_sentinel_position() {
        let out = drain(|state, out| state.use_item(4, None, out));
        assert_eq!(out.len(), 2, "held-item sync then the use");
        assert!(matches!(
            out[0],
            C2SPlay::HeldItemChange(c2s::HeldItemChange { slot_id: 4 })
        ));
        match &out[1] {
            C2SPlay::PlayerBlockPlacement(packet) => {
                assert_eq!(packet.position, BlockPos::new(-1, -1, -1));
                assert_eq!(packet.placed_block_direction, 255);
            }
            other => panic!("{other:?}"),
        }
    }

    /// java: `onStoppedUsingItem` (`:576`) — `syncCurrentPlayItem()`, then
    /// `C07PacketPlayerDigging(RELEASE_USE_ITEM, BlockPos.ORIGIN, EnumFacing.DOWN)`. This is
    /// the packet a bow, a piece of food and a potion all need and none of them ever got.
    #[test]
    fn release_use_item_on_initial_slot_zero_sends_no_redundant_sync() {
        let out = drain(|state, out| state.release_use_item(0, out));
        assert_eq!(out.len(), 1);
        match &out[0] {
            C2SPlay::PlayerDigging(packet) => {
                assert_eq!(packet.status, c2s::PlayerDiggingStatus::ReleaseUseItem);
                assert_eq!(packet.position, BlockPos::new(0, 0, 0), "BlockPos.ORIGIN");
                assert_eq!(packet.face, Facing::Down);
            }
            other => panic!("{other:?}"),
        }
    }

    /// The sync is suppressed once the server knows the slot, so a held bow released after a
    /// placement does not put a redundant `C09` on the wire.
    #[test]
    fn a_release_after_a_placement_on_the_same_slot_sends_no_second_held_item_change() {
        let mut state = OutboundState::new();
        let mut out = Vec::new();
        state.sync_held_item(2, &mut out);
        out.clear();
        state.release_use_item(2, &mut out);
        assert_eq!(out.len(), 1, "just the release: {out:?}");
        assert!(matches!(out[0], C2SPlay::PlayerDigging(_)));
    }

    #[test]
    fn armor_stand_interact_at_can_suppress_the_generic_interact_packet() {
        let suppressed = drain(|state, out| {
            state.use_entity_right_click(7, 0, [0.25, 1.0, -0.5], false, out);
        });
        assert_eq!(suppressed.len(), 1);
        assert!(matches!(
            suppressed[0],
            C2SPlay::UseEntity(c2s::UseEntity {
                action: c2s::UseEntityAction::InteractAt { .. },
                ..
            })
        ));

        let ordinary = drain(|state, out| {
            state.use_entity_right_click(7, 0, [0.25, 1.0, -0.5], true, out);
        });
        assert_eq!(ordinary.len(), 2);
        assert!(matches!(
            ordinary[1],
            C2SPlay::UseEntity(c2s::UseEntity {
                action: c2s::UseEntityAction::Interact,
                ..
            })
        ));
    }

    #[test]
    fn chat_is_truncated_at_a_hundred_utf16_units() {
        match chat_message(&"a".repeat(150)) {
            C2SPlay::ChatMessage(packet) => assert_eq!(packet.message.len(), 100),
            other => panic!("{other:?}"),
        }
        match chat_message("hello") {
            C2SPlay::ChatMessage(packet) => assert_eq!(packet.message, "hello"),
            other => panic!("{other:?}"),
        }
        // An astral character is two UTF-16 units, so 50 of them exactly fill the budget and
        // the 51st must be dropped whole rather than split.
        let emoji = "\u{1F600}".repeat(51);
        match chat_message(&emoji) {
            C2SPlay::ChatMessage(packet) => {
                assert_eq!(packet.message.chars().count(), 50);
            }
            other => panic!("{other:?}"),
        }

        // Java's substring(0, 100) keeps the high half of this final surrogate pair; Java's
        // UTF-8 encoder writes the orphan as `?`. A "never split a Rust char" mutant drops it.
        let split_pair = format!("{}\u{1F600}", "a".repeat(99));
        match chat_message(&split_pair) {
            C2SPlay::ChatMessage(packet) => {
                assert_eq!(packet.message, format!("{}?", "a".repeat(99)));
            }
            other => panic!("{other:?}"),
        }
    }

    /// With 256 Character Chat the cut moves to 256 UTF-16 units,
    /// with the same surrogate rules; the default path stays vanilla's 100.
    #[test]
    fn long_chat_is_truncated_at_two_hundred_fifty_six_utf16_units() {
        let message = |packet: C2SPlay| match packet {
            C2SPlay::ChatMessage(packet) => packet.message,
            other => panic!("{other:?}"),
        };
        assert_eq!(
            message(chat_message_with_limit(&"a".repeat(300), LONG_CHAT_LENGTH)).len(),
            256
        );
        assert_eq!(
            message(chat_message_with_limit(&"a".repeat(200), LONG_CHAT_LENGTH)).len(),
            200
        );
        assert_eq!(message(chat_message(&"a".repeat(200))).len(), 100);
        let split_pair = format!("{}\u{1F600}", "a".repeat(255));
        assert_eq!(
            message(chat_message_with_limit(&split_pair, LONG_CHAT_LENGTH)),
            format!("{}?", "a".repeat(255))
        );
        // The encoder writes all 256 characters: a VarInt length (256 = 0x80 0x02) and the bytes.
        let packet = c2s::ChatMessage {
            message: "b".repeat(256),
        };
        let mut buf = mc_protocol::buffer::PacketBuffer::new();
        mc_protocol::packet::Packet::write(&packet, &mut buf);
        let bytes = buf.into_inner();
        assert_eq!(&bytes[..2], &[0x80, 0x02]);
        assert_eq!(bytes.len(), 2 + 256);
    }

    #[test]
    fn close_window_preserves_the_server_window_id() {
        match close_window(251) {
            C2SPlay::CloseWindow(packet) => assert_eq!(packet.window_id, 251),
            other => panic!("expected C0D, got {other:?}"),
        }
    }

    /// java: the brand payload is a protocol string, so it carries its own VarInt length.
    #[test]
    fn the_brand_message_is_a_length_prefixed_string() {
        match brand_message("vanilla") {
            C2SPlay::PluginMessage(packet) => {
                assert_eq!(packet.channel, "MC|Brand");
                assert_eq!(packet.data[0], 7, "VarInt length of \"vanilla\"");
                assert_eq!(&packet.data[1..], b"vanilla");
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn resource_pack_status_echoes_the_hash_and_selected_vanilla_ordinal() {
        let hash = "0123456789abcdef0123456789abcdef01234567";
        match resource_pack_status(hash, c2s::ResourcePackStatusResult::Accepted) {
            C2SPlay::ResourcePackStatus(packet) => {
                assert_eq!(packet.hash, hash);
                assert_eq!(packet.result, c2s::ResourcePackStatusResult::Accepted);
            }
            other => panic!("expected C19, got {other:?}"),
        }
    }

    #[test]
    fn book_payload_is_the_complete_item_stack_without_an_extra_length_prefix() {
        let stack = mc_protocol::ItemStack {
            item_id: 387,
            count: 1,
            damage: 0,
            nbt: None,
        };
        match book_payload("MC|BSign", &stack) {
            C2SPlay::PluginMessage(packet) => {
                assert_eq!(packet.channel, "MC|BSign");
                let mut payload = mc_protocol::PacketBuffer::from_slice(&packet.data);
                assert_eq!(
                    mc_protocol::ItemStack::read(&mut payload).unwrap(),
                    Some(stack)
                );
                assert!(payload.as_slice().is_empty());
            }
            other => panic!("{other:?}"),
        }
    }

    /// java: `TileEntityCommandBlock.func_145757_a` writes raw x/y/z ints, not a packed
    /// `BlockPos` long; the command itself is then a VarInt-length-prefixed string.
    #[test]
    fn command_block_payload_uses_mode_zero_and_three_raw_coordinates() {
        let target = CommandBlockTarget::Block {
            position: BlockPos::new(-12, 255, 34),
        };
        let C2SPlay::PluginMessage(packet) = command_block_payload(target, "say hi", false) else {
            panic!("expected MC|AdvCdm plugin payload");
        };
        assert_eq!(packet.channel, "MC|AdvCdm");
        let mut payload = mc_protocol::PacketBuffer::from_slice(&packet.data);
        assert_eq!(payload.read_u8().unwrap(), 0);
        assert_eq!(payload.read_i32().unwrap(), -12);
        assert_eq!(payload.read_i32().unwrap(), 255);
        assert_eq!(payload.read_i32().unwrap(), 34);
        assert_eq!(payload.read_string(32767).unwrap(), "say hi");
        assert!(!payload.read_bool().unwrap());
        assert!(payload.as_slice().is_empty());
        assert_eq!(packet.data.len(), 1 + 12 + 1 + 6 + 1);
    }

    /// java: `EntityMinecartCommandBlock.func_145757_a` writes mode `1` and one entity id.
    #[test]
    fn command_minecart_payload_uses_mode_one_and_entity_id() {
        let target = CommandBlockTarget::Minecart { entity_id: -7 };
        let C2SPlay::PluginMessage(packet) = command_block_payload(target, "", true) else {
            panic!("expected MC|AdvCdm plugin payload");
        };
        assert_eq!(packet.channel, "MC|AdvCdm");
        let mut payload = mc_protocol::PacketBuffer::from_slice(&packet.data);
        assert_eq!(payload.read_u8().unwrap(), 1);
        assert_eq!(payload.read_i32().unwrap(), -7);
        assert_eq!(payload.read_string(32767).unwrap(), "");
        assert!(payload.read_bool().unwrap());
        assert!(payload.as_slice().is_empty());
        assert_eq!(packet.data.len(), 1 + 4 + 1 + 1);
    }

    /// **Byte-exact C2S encodings against the real 1.8.9 classes.** The reference
    /// is `javaref/l4_outbound_ref.txt` (`generate-l4-outbound.ps1`): each line is a vanilla
    /// packet built with the values below and serialized by its own `writePacketData`. Every
    /// packet here is built through the same constructor the live client uses, so a changed
    /// field order, width, enum ordinal or flag bit fails with the exact bytes Java writes.
    #[test]
    fn the_outbound_constructors_write_vanillas_exact_bytes() {
        let reference: std::collections::HashMap<&str, (i32, &str)> =
            include_str!("../javaref/l4_outbound_ref.txt")
                .lines()
                .filter_map(|line| line.strip_prefix("pkt "))
                .map(|line| {
                    let mut fields = line.split(' ');
                    let name = fields.next().expect("name");
                    let id = fields.next().expect("id").parse().expect("decimal id");
                    (name, (id, fields.next().expect("payload")))
                })
                .collect();
        let only = |packets: Vec<C2SPlay>| -> C2SPlay {
            assert_eq!(packets.len(), 1, "{packets:?}");
            packets.into_iter().next().expect("one packet")
        };
        // `new PlayerCapabilities()` with `allowFlying` and `isFlying` set.
        let abilities = crate::sync::Abilities::from_flags(0x02 | 0x04, 0.05, 0.1);
        let interact =
            drain(|state, out| state.use_entity_right_click(150, 0, [0.0; 3], true, out))
                .pop()
                .expect("INTERACT");
        let cases = [
            (
                "c03_ground",
                movement_packet(Outbound::Ground { on_ground: true }),
            ),
            (
                "c03_air",
                movement_packet(Outbound::Ground { on_ground: false }),
            ),
            (
                "c04",
                movement_packet(Outbound::Position {
                    pos: DVec3::new(-1234.5678901234, 64.00000001, 0.1 + 0.2),
                    on_ground: true,
                }),
            ),
            (
                "c05",
                movement_packet(Outbound::Look {
                    yaw: -1080.25,
                    pitch: -90.0,
                    on_ground: false,
                }),
            ),
            (
                "c06",
                movement_packet(Outbound::PositionLook {
                    pos: DVec3::new(29_999_999.5, -64.0, -0.0),
                    yaw: 359.99997,
                    pitch: 12.345_678,
                    on_ground: true,
                }),
            ),
            (
                "c0b_start_sneak",
                entity_action(3, c2s::EntityActionType::StartSneaking),
            ),
            (
                "c0b_stop_sneak",
                entity_action(3, c2s::EntityActionType::StopSneaking),
            ),
            (
                "c0b_stop_sleep",
                C2SPlay::EntityAction(c2s::EntityAction::stop_sleeping(300)),
            ),
            (
                "c0b_start_sprint",
                entity_action(123_456, c2s::EntityActionType::StartSprinting),
            ),
            (
                "c0b_stop_sprint",
                entity_action(123_456, c2s::EntityActionType::StopSprinting),
            ),
            ("c0b_riding_jump", horse_jump(7, 1.0)),
            ("c0b_open_inventory", horse_inventory(-1)),
            ("c0c_walk", steer_vehicle(0.98, 0.98, false, false)),
            ("c0c_sneak_jump", steer_vehicle(-0.294, 0.0, true, true)),
            ("c0a", only(drain(|state, out| state.swing_arm(out)))),
            (
                "c09",
                only(drain(|state, out| state.sync_held_item(8, out))),
            ),
            (
                "c02_attack",
                only(drain(|state, out| state.attack_entity(150, 0, out))),
            ),
            ("c02_interact", interact),
            (
                "c02_interact_at",
                only(drain(|state, out| {
                    state.use_entity_right_click(150, 0, [0.25, 1.5, -0.125], false, out);
                })),
            ),
            (
                "c07_start",
                only(drain(|state, out| {
                    state.start_digging(BlockPos::new(-5, 70, 1_234_567), Facing::East, out);
                })),
            ),
            (
                "c07_release",
                only(drain(|state, out| state.release_use_item(0, out))),
            ),
            (
                "c08_place",
                only(drain(|state, out| {
                    state.place_block(
                        BlockPos::new(10, 64, -3),
                        Facing::Up,
                        DVec3::new(0.5, 1.0, 0.25),
                        None,
                        0,
                        out,
                    );
                })),
            ),
            (
                "c08_use",
                only(drain(|state, out| state.use_item(0, None, out))),
            ),
            ("c13", player_abilities(&abilities)),
            (
                "c0f",
                C2SPlay::ConfirmTransaction(c2s::ConfirmTransaction {
                    window_id: 0,
                    action_id: -12345,
                    accepted: true,
                }),
            ),
            (
                "c00",
                C2SPlay::KeepAlive(c2s::KeepAlive {
                    key: 0xDEAD_BEEF_u32 as i32,
                }),
            ),
            ("c0d", close_window(0)),
            ("c16", perform_respawn()),
            (
                "c15",
                client_settings("en_US", 12, c2s::ChatVisibility::Full, true, 127),
            ),
        ];
        assert_eq!(
            cases.len(),
            reference.len(),
            "every reference packet is checked"
        );
        for (name, packet) in cases {
            let (id, payload) = reference[name];
            let mut buf = mc_protocol::PacketBuffer::new();
            packet.write(&mut buf);
            let hex: String = buf.as_slice().iter().map(|b| format!("{b:02x}")).collect();
            assert_eq!(packet.id(), id, "{name}: packet id");
            assert_eq!(
                if hex.is_empty() { "-" } else { hex.as_str() },
                payload,
                "{name}: {packet:?}"
            );
        }
    }

    /// **Every look the C05/C06 report is one a vanilla client reaches.** The
    /// reference (`javaref/l4_outbound_ref.txt`) replays random integer `MouseHelper` deltas
    /// through `EntityRenderer`'s verbatim mouse and `updateRenderer` blocks into the real
    /// `Entity.setAngles` and `MouseFilter`, at the user's sensitivity among others, with the
    /// smooth camera off, on, and toggled (as F8 or the Zoom mod do), inverted or not. This
    /// composes the live frame path exactly as `Client::frame` does (`look_delta_raw` with no
    /// invert, the smooth filter, invert, then `turn_mouse` on the predictor's look) and the
    /// tick's `SmoothCamera::tick`, and requires bit-identical yaw and pitch after every frame:
    /// anything off vanilla's sensitivity grid or its float/double rounding fails here.
    #[test]
    fn mouse_look_lands_on_vanillas_exact_rotations() {
        use mc_render::input::CursorMode;

        let hex = |field: &str| u32::from_str_radix(field, 16).expect("hex bits");
        let mut frames = 0usize;
        let mut state: Option<(f32, bool, mc_render::InputState, mc_render::SmoothCamera)> = None;
        let mut smooth_on = false;
        let mut look = mc_render::Camera::default();
        for line in include_str!("../javaref/l4_outbound_ref.txt").lines() {
            let fields: Vec<&str> = line.split(' ').collect();
            match fields[0] {
                "case" => {
                    let mut input = mc_render::InputState::new();
                    input.request_cursor(CursorMode::Grabbed);
                    state = Some((
                        f32::from_bits(hex(fields[2])),
                        fields[3] == "1",
                        input,
                        mc_render::SmoothCamera::default(),
                    ));
                    look = mc_render::Camera::default();
                    look.yaw = 0.0;
                    look.pitch = 0.0;
                }
                "s" => smooth_on = fields[1] == "1",
                "t" => {
                    let (sensitivity, _, _, smooth) = state.as_mut().expect("case first");
                    smooth.tick(*sensitivity, smooth_on);
                }
                "f" => {
                    let (sensitivity, invert, input, smooth) = state.as_mut().expect("case");
                    let dx: i32 = fields[1].parse().expect("dx");
                    let dy: i32 = fields[2].parse().expect("dy");
                    let partial = f32::from_bits(hex(fields[3]));
                    // LWJGL's deltaY points up; winit's raw motion points down.
                    input.on_raw_motion(f64::from(dx), -f64::from(dy));
                    let (yaw, pitch) = input.look_delta_raw(*sensitivity, false);
                    let (yaw, pitch) = smooth.apply(yaw, pitch, partial, smooth_on);
                    let pitch = pitch * if *invert { -1.0 } else { 1.0 };
                    look.turn_mouse(yaw, pitch);
                    assert_eq!(
                        (look.yaw.to_bits(), look.pitch.to_bits()),
                        (hex(fields[4]), hex(fields[5])),
                        "frame {frames}: {line} gave yaw {} pitch {}",
                        look.yaw,
                        look.pitch
                    );
                    frames += 1;
                }
                _ => {}
            }
        }
        assert!(frames > 10_000, "the whole trace ran: {frames}");
    }

    /// java: `handlePlayerPosLook` replies with `onGround = false` regardless of the real state.
    #[test]
    fn a_position_confirmation_always_claims_it_is_airborne() {
        match confirm_position(DVec3::new(1.0, 2.0, 3.0), 90.0, -10.0) {
            C2SPlay::PlayerPositionAndLook(packet) => {
                assert!(!packet.on_ground);
                assert_eq!(packet.yaw, 90.0);
            }
            other => panic!("{other:?}"),
        }
    }
}
