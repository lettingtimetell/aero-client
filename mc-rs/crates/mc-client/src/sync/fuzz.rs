//! Structure-aware random clientbound packets for crash testing.
//!
//! java: a vanilla handler exception on the main thread is caught and logged by `Util.runTask`
//! ("Error executing task") and the game carries on; on the Netty thread it closes the channel
//! through `exceptionCaught`. A Rust panic in either place is a crash instead, so every handler
//! must survive any value the wire can deliver.
//!
//! Random bytes almost never decode into packets that reach deep handler code, so this builds
//! decoded [`S2CPlay`] values directly. Every numeric field comes from a boundary-heavy
//! distribution (zero, +/-1, the type's extremes, NaN and the infinities) and entity ids come
//! mostly from a small shared pool, so spawns, moves, metadata, attaches, effects and removals
//! keep landing on the same live entities. Each candidate then goes through a `write` ->
//! `S2CPlay::read` round trip ([`wire_reachable`]): only values the real decoder can produce
//! reach a handler, so a failure here is a crash a server can trigger, not a test artefact.

use mc_math::{BlockPos, Facing};
use mc_nbt::{Nbt, NbtCompound};
use mc_protocol::chunk::{ChunkBulkColumn, ChunkData, MapChunkBulk};
use mc_protocol::play::{s2c, S2CPlay};
use mc_protocol::{Angle, ItemStack, MetadataEntry, MetadataValue, PacketBuffer};
use uuid::Uuid;

/// A deterministic xorshift64* stream: every failure reproduces from its seed.
pub(crate) struct PacketFuzzer {
    state: u64,
    /// The local player's entity id, so a share of entity-targeted packets hit it.
    pub(crate) local_entity_id: i32,
    /// When the client also ticks, keep the values that feed the local player's motion
    /// (explosion knockback, ability and attribute speeds) finite and small, with NaN still
    /// allowed. A huge one is a faithful freeze, not a crash: vanilla's `moveEntity` sweeps
    /// `getCollidingBoundingBoxes` over the same `Integer.MAX_VALUE`-saturated box.
    pub(crate) physical: bool,
}

/// Re-encode and decode `packet`, keeping it only if the real decoder accepts its bytes in
/// full. This normalizes the generator's output to exactly what a server can deliver.
pub(crate) fn wire_reachable(packet: S2CPlay) -> Option<S2CPlay> {
    let mut buf = PacketBuffer::new();
    packet.write(&mut buf);
    buf.finish_write().ok()?;
    let mut read = PacketBuffer::from_slice(buf.as_slice());
    let decoded = S2CPlay::read(packet.id(), &mut read).ok()?;
    read.is_empty().then_some(decoded)
}

impl PacketFuzzer {
    pub(crate) fn new(seed: u64, local_entity_id: i32) -> Self {
        Self {
            state: seed | 1,
            local_entity_id,
            physical: false,
        }
    }

    /// A motion-feeding `f32`: anything in dispatch-only mode, bounded in [`Self::physical`].
    fn motion_f32(&mut self) -> f32 {
        if !self.physical {
            return self.float();
        }
        match self.below(8) {
            0 => f32::NAN,
            1 => 0.0,
            _ => (self.below(161) as f32 - 80.0) / 10.0,
        }
    }

    /// An S2B value. A huge finite rain level is a faithful freeze (vanilla's
    /// `addRainParticles` loops `(int)(100 * f * f)` times), so [`Self::physical`] mode keeps
    /// finite levels small while still sending the infinities and NaN.
    fn level_f32(&mut self) -> f32 {
        if !self.physical {
            return self.float();
        }
        match self.below(6) {
            0 => f32::INFINITY,
            1 => f32::NEG_INFINITY,
            2 => f32::NAN,
            _ => self.below(301) as f32 / 100.0,
        }
    }

    /// A motion-feeding `f64`; see [`Self::motion_f32`].
    fn motion_f64(&mut self) -> f64 {
        if !self.physical {
            return self.double();
        }
        f64::from(self.motion_f32())
    }

    /// An S08 coordinate. In [`Self::physical`] mode it stays inside the world border's range
    /// (plus NaN): beyond `i32` every tick-time block-coordinate sum wraps in release builds
    /// exactly as Java's `int` does, so the remaining checks there are debug-build overflow
    /// assertions in the light engine, not crashes a player can hit.
    fn teleport(&mut self) -> f64 {
        if !self.physical {
            return self.double();
        }
        match self.below(6) {
            0 => f64::NAN,
            1 => [29_999_999.5, -29_999_999.5][self.below(2) as usize],
            _ => (self.below(2001) as f64 - 1000.0) / 4.0,
        }
    }

    fn next(&mut self) -> u64 {
        self.state ^= self.state >> 12;
        self.state ^= self.state << 25;
        self.state ^= self.state >> 27;
        self.state.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }

    fn one_in(&mut self, n: u64) -> bool {
        self.below(n) == 0
    }

    fn flag(&mut self) -> bool {
        self.next() & 1 == 1
    }

    fn int(&mut self) -> i32 {
        match self.below(12) {
            0 => 0,
            1 => 1,
            2 => -1,
            3 => i32::MIN,
            4 => i32::MAX,
            5 => i32::MIN + 1,
            6 => i32::MAX - 1,
            7 | 8 => self.below(129) as i32 - 64,
            9 => self.below(65_536) as i32 - 32_768,
            _ => self.next() as i32,
        }
    }

    fn short(&mut self) -> i16 {
        match self.below(8) {
            0 => 0,
            1 => i16::MIN,
            2 => i16::MAX,
            3 => -1,
            4 | 5 => self.below(129) as i16 - 64,
            _ => self.next() as i16,
        }
    }

    fn byte(&mut self) -> i8 {
        match self.below(6) {
            0 => 0,
            1 => i8::MIN,
            2 => i8::MAX,
            3 => -1,
            _ => self.next() as i8,
        }
    }

    fn ubyte(&mut self) -> u8 {
        self.byte() as u8
    }

    fn long(&mut self) -> i64 {
        match self.below(8) {
            0 => 0,
            1 => i64::MIN,
            2 => i64::MAX,
            3 => -1,
            4 => i64::from(self.int()),
            _ => self.next() as i64,
        }
    }

    fn float(&mut self) -> f32 {
        match self.below(14) {
            0 => 0.0,
            1 => -0.0,
            2 => f32::NAN,
            3 => f32::INFINITY,
            4 => f32::NEG_INFINITY,
            5 => f32::MAX,
            6 => f32::MIN,
            7 => f32::MIN_POSITIVE,
            8 => 1.0,
            9 => -1.0,
            10 => f32::from_bits(self.next() as u32),
            _ => (self.below(2001) as f32 - 1000.0) / 8.0,
        }
    }

    fn double(&mut self) -> f64 {
        match self.below(14) {
            0 => 0.0,
            1 => -0.0,
            2 => f64::NAN,
            3 => f64::INFINITY,
            4 => f64::NEG_INFINITY,
            5 => f64::MAX,
            6 => f64::MIN,
            7 => 3.0e7,
            8 => -3.0e7,
            9 => 1.0e300,
            10 => f64::from_bits(self.next()),
            _ => (self.below(20_001) as f64 - 10_000.0) / 16.0,
        }
    }

    fn angle(&mut self) -> Angle {
        Angle(self.next() as u8)
    }

    /// Mostly a small shared pool, sometimes the local player, sometimes any `int`.
    fn entity_id(&mut self) -> i32 {
        match self.below(10) {
            0 => self.local_entity_id,
            1 | 2 => self.int(),
            _ => self.below(12) as i32,
        }
    }

    /// Fixed-point (x32) coordinates: mostly near the loaded chunks, sometimes anything.
    fn fixed(&mut self) -> i32 {
        if self.one_in(5) {
            self.int()
        } else {
            self.below(32 * 96) as i32 - 32 * 48
        }
    }

    /// The packed position decodes from any `i64`, so draw from that full range sometimes.
    fn block_pos(&mut self) -> BlockPos {
        if self.one_in(5) {
            BlockPos::from_long(self.next() as i64)
        } else {
            BlockPos::new(
                self.below(64) as i32 - 32,
                self.below(300) as i32 - 20,
                self.below(64) as i32 - 32,
            )
        }
    }

    fn text(&mut self, max_chars: usize) -> String {
        const PIECES: &[&str] = &[
            "a",
            "Z",
            "0",
            " ",
            "\u{a7}",
            "\u{a7}c",
            "\u{a7}k",
            "\u{a7}r",
            "\u{1f600}",
            "\u{fffd}",
            "%s",
            "%1$s",
            "%%",
            "\n",
            "\"",
            "\\",
            "{",
            "[",
            "null",
            "\u{0}",
        ];
        let len = self.below(max_chars as u64 + 1) as usize;
        let mut out = String::new();
        while out.encode_utf16().count() < len {
            let piece = PIECES[self.below(PIECES.len() as u64) as usize];
            if out.encode_utf16().count() + piece.encode_utf16().count() > max_chars {
                break;
            }
            out.push_str(piece);
        }
        out
    }

    /// A chat component Gson accepts (including the spellings it resolves to null).
    fn json(&mut self) -> String {
        match self.below(12) {
            0 => "null".to_owned(),
            1 => String::new(),
            2 => "[]".to_owned(),
            3 => format!("{:?}", self.text(40)),
            4 => r#"{"translate":"chat.type.text","with":["a",{"text":"b"}]}"#.to_owned(),
            5 => r#"{"translate":"%s %s %1$s %%","with":[[],"x"]}"#.to_owned(),
            6 => r#"{"text":"","extra":[{"text":"x","color":"red","obfuscated":true}]}"#.to_owned(),
            7 => r#"{"score":{"name":"*","objective":"o"}}"#.to_owned(),
            8 => r#"{"selector":"@p"}"#.to_owned(),
            9 => format!(r#"{{"text":{:?},"bold":true}}"#, self.text(60)),
            10 => r#"[{"text":"a"},"b",{"translate":"missing.key.%s"}]"#.to_owned(),
            _ => format!("{}", self.int()),
        }
    }

    fn uuid(&mut self) -> Uuid {
        if self.one_in(4) {
            Uuid::from_u64_pair(self.next(), self.next())
        } else {
            Uuid::from_u128(u128::from(self.below(6)))
        }
    }

    fn nbt(&mut self) -> Option<NbtCompound> {
        if self.below(3) != 0 {
            return None;
        }
        let mut tag = NbtCompound::new();
        if self.flag() {
            let mut display = NbtCompound::new();
            display.insert("Name", Nbt::String(self.text(30)));
            if self.flag() {
                display.insert(
                    "Lore",
                    Nbt::list(vec![Nbt::String(self.text(20)), Nbt::String(self.text(5))]),
                );
            }
            if self.flag() {
                display.insert("color", Nbt::Int(self.int()));
            }
            tag.insert("display", Nbt::Compound(display));
        }
        if self.flag() {
            let mut enchantment = NbtCompound::new();
            enchantment.insert("id", Nbt::Short(self.short()));
            enchantment.insert("lvl", Nbt::Short(self.short()));
            tag.insert("ench", Nbt::list(vec![Nbt::Compound(enchantment)]));
        }
        if self.flag() {
            tag.insert("HideFlags", Nbt::Int(self.int()));
        }
        if self.one_in(3) {
            tag.insert("SkullOwner", Nbt::String(self.text(16)));
        }
        if self.one_in(3) {
            let mut explosion = NbtCompound::new();
            explosion.insert("Type", Nbt::Byte(self.byte()));
            explosion.insert("Colors", Nbt::IntArray(vec![self.int(), self.int()]));
            let mut fireworks = NbtCompound::new();
            fireworks.insert("Explosions", Nbt::list(vec![Nbt::Compound(explosion)]));
            fireworks.insert("Flight", Nbt::Byte(self.byte()));
            tag.insert("Fireworks", Nbt::Compound(fireworks));
        }
        if self.one_in(4) {
            tag.insert(
                "pages",
                Nbt::list(vec![Nbt::String(self.json()), Nbt::String(self.text(50))]),
            );
            tag.insert("title", Nbt::String(self.text(32)));
            tag.insert("author", Nbt::String(self.text(16)));
        }
        if self.one_in(4) {
            let mut potion = NbtCompound::new();
            potion.insert("Id", Nbt::Byte(self.byte()));
            potion.insert("Amplifier", Nbt::Byte(self.byte()));
            potion.insert("Duration", Nbt::Int(self.int()));
            tag.insert(
                "CustomPotionEffects",
                Nbt::list(vec![Nbt::Compound(potion)]),
            );
        }
        Some(tag)
    }

    fn item(&mut self) -> Option<ItemStack> {
        if self.one_in(4) {
            return None;
        }
        let item_id = match self.below(5) {
            0 => self.below(198) as i16,
            1 | 2 => 256 + self.below(200) as i16,
            3 => 2256 + self.below(12) as i16,
            _ => self.short().max(0),
        };
        Some(ItemStack {
            item_id,
            count: i32::from(self.byte()),
            damage: self.short().max(0),
            nbt: self.nbt(),
        })
    }

    fn metadata(&mut self) -> Vec<MetadataEntry> {
        let count = self.below(7);
        (0..count)
            .map(|_| {
                let index = self.below(32) as u8;
                let value = match self.below(8) {
                    0 => MetadataValue::Byte(self.byte()),
                    1 => MetadataValue::Short(self.short()),
                    2 => MetadataValue::Int(self.int()),
                    3 => MetadataValue::Float(self.float()),
                    4 => MetadataValue::String(self.text(64)),
                    5 => MetadataValue::ItemStack(self.item()),
                    6 => MetadataValue::BlockPos(BlockPos::new(self.int(), self.int(), self.int())),
                    _ => MetadataValue::Rotations(self.float(), self.float(), self.float()),
                };
                MetadataEntry { index, value }
            })
            .collect()
    }

    fn chunk_payload(&mut self, mask: u16, sky: bool, ground_up: bool) -> Vec<u8> {
        let len = mc_world::expected_chunk_data_len(mask, sky, ground_up);
        let mut data = vec![0u8; len];
        // Mostly real block ids, some arbitrary (illegal ids read back as air), random light
        // and biome bytes (biome ids above 39 are unregistered).
        for value in &mut data {
            *value = if self.one_in(3) { self.next() as u8 } else { 0 };
        }
        data
    }

    fn chunk_xz(&mut self) -> i32 {
        if self.one_in(20) {
            self.int()
        } else {
            self.below(5) as i32 - 2
        }
    }

    /// One candidate clientbound packet. The three Netty-thread packets the connection task
    /// consumes itself (KeepAlive, SetCompression, Disconnect) are never generated: they never
    /// reach the game thread.
    #[allow(clippy::too_many_lines)]
    pub(crate) fn candidate(&mut self) -> S2CPlay {
        match self.below(70) {
            0 if self.one_in(40) => S2CPlay::JoinGame(s2c::JoinGame {
                entity_id: self.local_entity_id,
                hardcore_mode: self.flag(),
                game_mode: self.ubyte() & !8,
                dimension: [-1, 0, 1, self.byte()][self.below(4) as usize],
                difficulty: self.ubyte(),
                max_players: self.ubyte(),
                level_type: ["default", "flat", "largeBiomes", "amplified", "zzz"]
                    [self.below(5) as usize]
                    .to_owned(),
                reduced_debug_info: self.flag(),
            }),
            1 => S2CPlay::Chat(s2c::Chat {
                component: self.json(),
                position: self.byte(),
            }),
            2 => S2CPlay::TimeUpdate(s2c::TimeUpdate {
                total_world_time: self.long(),
                world_time: self.long(),
            }),
            3 => S2CPlay::EntityEquipment(s2c::EntityEquipment {
                entity_id: self.entity_id(),
                equipment_slot: self.short(),
                item: self.item(),
            }),
            4 => S2CPlay::SpawnPosition(s2c::SpawnPosition {
                position: self.block_pos(),
            }),
            5 => S2CPlay::UpdateHealth(s2c::UpdateHealth {
                health: self.float(),
                food_level: self.int(),
                saturation_level: self.float(),
            }),
            6 if self.one_in(20) => S2CPlay::Respawn(s2c::Respawn {
                dimension: [-1, 0, 1, self.int()][self.below(4) as usize],
                difficulty: self.ubyte(),
                game_mode: self.ubyte(),
                level_type: "default".to_owned(),
            }),
            7 => S2CPlay::PlayerPositionLook(s2c::PlayerPositionLook {
                x: self.teleport(),
                y: self.teleport(),
                z: self.teleport(),
                yaw: self.float(),
                pitch: self.float(),
                flags: self.ubyte(),
            }),
            8 => S2CPlay::HeldItemChange(s2c::HeldItemChange { slot: self.byte() }),
            9 => S2CPlay::UseBed(s2c::UseBed {
                player_id: self.entity_id(),
                bed_position: self.block_pos(),
            }),
            10 => S2CPlay::Animation(s2c::Animation {
                entity_id: self.entity_id(),
                animation_type: self.ubyte(),
            }),
            11 => S2CPlay::SpawnPlayer(s2c::SpawnPlayer {
                entity_id: self.entity_id(),
                player_uuid: self.uuid(),
                x: self.fixed(),
                y: self.fixed(),
                z: self.fixed(),
                yaw: self.angle(),
                pitch: self.angle(),
                current_item: self.short(),
                metadata: self.metadata(),
            }),
            12 => S2CPlay::CollectItem(s2c::CollectItem {
                collected_entity_id: self.entity_id(),
                collector_entity_id: self.entity_id(),
            }),
            13 | 14 => {
                let data = self.int();
                S2CPlay::SpawnObject(s2c::SpawnObject {
                    entity_id: self.entity_id(),
                    object_type: [
                        1, 2, 10, 50, 51, 60, 61, 62, 63, 64, 65, 66, 70, 71, 72, 73, 75, 76, 77,
                        78, 90,
                    ][self.below(21) as usize],
                    x: self.fixed(),
                    y: self.fixed(),
                    z: self.fixed(),
                    pitch: self.angle(),
                    yaw: self.angle(),
                    data,
                    velocity: (data > 0).then(|| (self.short(), self.short(), self.short())),
                })
            }
            15 | 16 => S2CPlay::SpawnMob(s2c::SpawnMob {
                entity_id: self.entity_id(),
                entity_type: if self.one_in(4) {
                    self.ubyte()
                } else {
                    [
                        50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68,
                        90, 91, 92, 93, 94, 95, 96, 98, 99, 100, 101, 120, 30,
                    ][self.below(32) as usize]
                },
                x: self.fixed(),
                y: self.fixed(),
                z: self.fixed(),
                yaw: self.angle(),
                pitch: self.angle(),
                head_pitch: self.angle(),
                velocity_x: self.short(),
                velocity_y: self.short(),
                velocity_z: self.short(),
                metadata: self.metadata(),
            }),
            17 => S2CPlay::SpawnPainting(s2c::SpawnPainting {
                entity_id: self.entity_id(),
                title: ["Kebab", "SkullAndRoses", "Pointer", "Nope", ""][self.below(5) as usize]
                    .to_owned(),
                position: self.block_pos(),
                facing: [Facing::North, Facing::South, Facing::West, Facing::East]
                    [self.below(4) as usize],
            }),
            18 => S2CPlay::SpawnExperienceOrb(s2c::SpawnExperienceOrb {
                entity_id: self.entity_id(),
                x: self.fixed(),
                y: self.fixed(),
                z: self.fixed(),
                xp_value: self.short(),
            }),
            19 => S2CPlay::EntityVelocity(s2c::EntityVelocity {
                entity_id: self.entity_id(),
                motion_x: self.short(),
                motion_y: self.short(),
                motion_z: self.short(),
            }),
            20 => S2CPlay::DestroyEntities(s2c::DestroyEntities {
                entity_ids: (0..self.below(4)).map(|_| self.entity_id()).collect(),
            }),
            21 => S2CPlay::Entity(s2c::Entity {
                entity_id: self.entity_id(),
            }),
            22 => S2CPlay::EntityRelMove(s2c::EntityRelMove {
                entity_id: self.entity_id(),
                delta_x: self.byte(),
                delta_y: self.byte(),
                delta_z: self.byte(),
                on_ground: self.flag(),
            }),
            23 => S2CPlay::EntityLook(s2c::EntityLook {
                entity_id: self.entity_id(),
                yaw: self.angle(),
                pitch: self.angle(),
                on_ground: self.flag(),
            }),
            24 => S2CPlay::EntityLookAndRelMove(s2c::EntityLookAndRelMove {
                entity_id: self.entity_id(),
                delta_x: self.byte(),
                delta_y: self.byte(),
                delta_z: self.byte(),
                yaw: self.angle(),
                pitch: self.angle(),
                on_ground: self.flag(),
            }),
            25 => S2CPlay::EntityTeleport(s2c::EntityTeleport {
                entity_id: self.entity_id(),
                x: self.fixed(),
                y: self.fixed(),
                z: self.fixed(),
                yaw: self.angle(),
                pitch: self.angle(),
                on_ground: self.flag(),
            }),
            26 => S2CPlay::EntityHeadLook(s2c::EntityHeadLook {
                entity_id: self.entity_id(),
                yaw: self.angle(),
            }),
            27 => S2CPlay::EntityStatus(s2c::EntityStatus {
                entity_id: self.entity_id(),
                opcode: if self.flag() {
                    self.below(24) as i8
                } else {
                    self.byte()
                },
            }),
            28 => S2CPlay::EntityAttach(s2c::EntityAttach {
                entity_id: self.entity_id(),
                vehicle_entity_id: if self.one_in(3) { -1 } else { self.entity_id() },
                attach_type: match self.below(3) {
                    0 => s2c::AttachType::Vehicle,
                    1 => s2c::AttachType::Leash,
                    _ => s2c::AttachType::Unknown(self.ubyte()),
                },
            }),
            29 | 30 => S2CPlay::EntityMetadata(s2c::EntityMetadata {
                entity_id: self.entity_id(),
                metadata: self.metadata(),
            }),
            31 => S2CPlay::EntityEffect(s2c::EntityEffect {
                entity_id: self.entity_id(),
                effect_id: self.byte(),
                amplifier: self.byte(),
                duration: self.int(),
                show_particles: self.flag(),
            }),
            32 => S2CPlay::RemoveEntityEffect(s2c::RemoveEntityEffect {
                entity_id: self.entity_id(),
                effect_id: self.ubyte(),
            }),
            33 => S2CPlay::SetExperience(s2c::SetExperience {
                experience_bar: self.float(),
                level: self.int(),
                total_experience: self.int(),
            }),
            34 => S2CPlay::EntityProperties(s2c::EntityProperties {
                entity_id: self.entity_id(),
                properties: (0..self.below(3))
                    .map(|_| s2c::AttributePropertySnapshot {
                        key: [
                            "generic.maxHealth",
                            "generic.movementSpeed",
                            "generic.attackDamage",
                            "generic.knockbackResistance",
                            "generic.followRange",
                            "horse.jumpStrength",
                            "zombie.spawnReinforcements",
                            "x",
                        ][self.below(8) as usize]
                            .to_owned(),
                        base_value: self.motion_f64(),
                        modifiers: (0..self.below(3))
                            .map(|_| s2c::AttributeModifierSnapshot {
                                uuid: self.uuid(),
                                amount: self.motion_f64(),
                                operation: match self.below(3) {
                                    0 => s2c::AttributeModifierOperation::Add,
                                    1 => s2c::AttributeModifierOperation::MultiplyBase,
                                    _ => s2c::AttributeModifierOperation::Multiply,
                                },
                            })
                            .collect(),
                    })
                    .collect(),
            }),
            35 | 36 => {
                let ground_up = self.flag();
                let mask = if self.one_in(5) {
                    0
                } else {
                    self.next() as u16
                };
                let sky = self.flag();
                S2CPlay::ChunkData(ChunkData {
                    chunk_x: self.chunk_xz(),
                    chunk_z: self.chunk_xz(),
                    ground_up_continuous: ground_up,
                    primary_bit_mask: mask,
                    data: self.chunk_payload(mask, sky, ground_up),
                })
            }
            37 => S2CPlay::MultiBlockChange(s2c::MultiBlockChange {
                chunk_x: self.chunk_xz(),
                chunk_z: self.chunk_xz(),
                changes: (0..self.below(5))
                    .map(|_| s2c::MultiBlockChangeEntry {
                        rel_x: self.below(16) as u8,
                        y: self.ubyte(),
                        rel_z: self.below(16) as u8,
                        block_state_id: self.block_state_id(),
                    })
                    .collect(),
            }),
            38 | 39 => S2CPlay::BlockChange(s2c::BlockChange {
                position: self.block_pos(),
                block_state_id: self.block_state_id(),
            }),
            40 => S2CPlay::BlockAction(s2c::BlockAction {
                position: self.block_pos(),
                data1: self.ubyte(),
                data2: self.ubyte(),
                block_id: if self.flag() {
                    [25, 29, 33, 54, 130, 146, 138][self.below(7) as usize]
                } else {
                    self.int()
                },
            }),
            41 => S2CPlay::BlockBreakAnimation(s2c::BlockBreakAnimation {
                breaker_id: self.entity_id(),
                position: self.block_pos(),
                progress: self.ubyte(),
            }),
            42 => {
                let sky = self.flag();
                S2CPlay::MapChunkBulk(MapChunkBulk {
                    has_sky_light: sky,
                    columns: (0..self.below(3))
                        .map(|_| {
                            let mask = self.next() as u16;
                            ChunkBulkColumn {
                                chunk_x: self.chunk_xz(),
                                chunk_z: self.chunk_xz(),
                                primary_bit_mask: mask,
                                data: self.chunk_payload(mask, sky, true),
                            }
                        })
                        .collect(),
                })
            }
            43 => S2CPlay::Explosion(s2c::Explosion {
                x: f64::from(self.float()),
                y: f64::from(self.float()),
                z: f64::from(self.float()),
                strength: self.float(),
                affected_blocks: (0..self.below(6))
                    .map(|_| s2c::ExplosionAffectedBlock {
                        dx: self.byte(),
                        dy: self.byte(),
                        dz: self.byte(),
                    })
                    .collect(),
                motion_x: self.motion_f32(),
                motion_y: self.motion_f32(),
                motion_z: self.motion_f32(),
            }),
            44 => S2CPlay::Effect(s2c::Effect {
                effect_id: if self.flag() {
                    [
                        1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011,
                        1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023,
                        2000, 2001, 2002, 2003, 2004, 2005,
                    ][self.below(30) as usize]
                } else {
                    self.int()
                },
                position: self.block_pos(),
                data: self.int(),
                disable_relative_volume: self.flag(),
            }),
            45 => S2CPlay::SoundEffect(s2c::SoundEffect {
                sound_name: ["random.click", "note.harp", "mob.zombie.say", "", "x.y.z"]
                    [self.below(5) as usize]
                    .to_owned(),
                x_fixed: self.int(),
                y_fixed: self.int(),
                z_fixed: self.int(),
                volume: self.float(),
                pitch_quantized: self.ubyte(),
            }),
            46 => {
                let particle_id = if self.flag() {
                    self.below(42) as i32
                } else {
                    self.int()
                };
                S2CPlay::Particles(s2c::Particles {
                    particle_id,
                    long_distance: self.flag(),
                    x: self.float(),
                    y: self.float(),
                    z: self.float(),
                    // Offsets and speed become particle velocities. Some constructors (crit,
                    // magic crit) call `onUpdate` at once, so a huge one sweeps an
                    // `Integer.MAX_VALUE`-wide collision box in vanilla as well (a freeze).
                    offset_x: self.motion_f32(),
                    offset_y: self.motion_f32(),
                    offset_z: self.motion_f32(),
                    particle_speed: self.motion_f32(),
                    // A huge count is a faithful freeze, not a crash: handleParticles loops
                    // `for (i = 0; i < count; ++i)` and so does mc-render's burst. Keep counts
                    // small (negative ones included, which spawn nothing in both).
                    particle_count: match self.below(4) {
                        0 => 0,
                        1 => [i32::MIN, -1][self.below(2) as usize],
                        _ => self.below(64) as i32,
                    },
                    // The S2C writer indexes the particle's fixed arity (at most 2); the round
                    // trip then keeps exactly what the reader consumes.
                    extra_data: (0..3).map(|_| self.int()).collect(),
                })
            }
            47 => S2CPlay::ChangeGameState(s2c::ChangeGameState {
                reason: match self.below(11) {
                    0 => s2c::GameStateReason::InvalidBed,
                    1 => s2c::GameStateReason::EndRaining,
                    2 => s2c::GameStateReason::BeginRaining,
                    3 => s2c::GameStateReason::ChangeGameMode,
                    4 if self.one_in(10) => s2c::GameStateReason::EnterCredits,
                    5 => s2c::GameStateReason::DemoMessage,
                    6 => s2c::GameStateReason::ArrowHittingPlayer,
                    7 => s2c::GameStateReason::RainLevelChange,
                    8 => s2c::GameStateReason::ThunderLevelChange,
                    9 => s2c::GameStateReason::PlayMobAppearance,
                    _ => s2c::GameStateReason::Other(self.ubyte()),
                },
                value: self.level_f32(),
            }),
            48 => S2CPlay::SpawnGlobalEntity(s2c::SpawnGlobalEntity {
                entity_id: self.entity_id(),
                entity_type: self.byte(),
                x_fixed: self.int(),
                y_fixed: self.int(),
                z_fixed: self.int(),
            }),
            49 => {
                let inventory_type = [
                    "minecraft:chest",
                    "minecraft:container",
                    "minecraft:crafting_table",
                    "minecraft:furnace",
                    "minecraft:dispenser",
                    "minecraft:enchanting_table",
                    "minecraft:brewing_stand",
                    "minecraft:villager",
                    "minecraft:beacon",
                    "minecraft:anvil",
                    "minecraft:hopper",
                    "minecraft:dropper",
                    "EntityHorse",
                    "minecraft:zzz",
                ][self.below(14) as usize]
                    .to_owned();
                let entity_id = (inventory_type == "EntityHorse").then(|| self.entity_id());
                S2CPlay::OpenWindow(s2c::OpenWindow {
                    window_id: self.ubyte(),
                    inventory_type,
                    window_title: self.json(),
                    slot_count: self.ubyte(),
                    entity_id,
                })
            }
            50 => S2CPlay::CloseWindow(s2c::CloseWindow {
                window_id: self.ubyte(),
            }),
            51 | 52 => S2CPlay::SetSlot(s2c::SetSlot {
                window_id: [0, -1, -2, 1, self.byte()][self.below(5) as usize],
                slot: if self.flag() {
                    self.below(46) as i16
                } else {
                    self.short()
                },
                item: self.item(),
            }),
            53 => S2CPlay::WindowItems(s2c::WindowItems {
                window_id: [0, 1, 2, self.ubyte()][self.below(4) as usize],
                items: (0..[45, 0, 3, 90, self.below(120)][self.below(5) as usize])
                    .map(|_| self.item())
                    .collect(),
            }),
            54 => S2CPlay::WindowProperty(s2c::WindowProperty {
                window_id: [0, 1, self.ubyte()][self.below(3) as usize],
                var_index: self.short(),
                var_value: self.short(),
            }),
            55 => S2CPlay::ConfirmTransaction(s2c::ConfirmTransaction {
                window_id: [0, 1, self.ubyte()][self.below(3) as usize],
                action_number: self.short(),
                accepted: self.flag(),
            }),
            56 => S2CPlay::UpdateSign(s2c::UpdateSign {
                position: self.block_pos(),
                lines: [self.json(), self.json(), self.json(), self.json()],
            }),
            57 => {
                let width = self.ubyte();
                S2CPlay::Maps(s2c::Maps {
                    map_id: if self.flag() {
                        self.below(4) as i32
                    } else {
                        self.int()
                    },
                    scale: self.byte(),
                    icons: (0..self.below(4))
                        .map(|_| s2c::MapIcon {
                            icon_type: self.below(16) as u8,
                            x: self.byte(),
                            z: self.byte(),
                            direction: self.below(16) as u8,
                        })
                        .collect(),
                    update: (width > 0).then(|| {
                        let height = self.ubyte();
                        let len = if self.flag() {
                            usize::from(width) * usize::from(height)
                        } else {
                            self.below(300) as usize
                        };
                        s2c::MapPixelUpdate {
                            width,
                            height,
                            min_x: self.ubyte(),
                            min_y: self.ubyte(),
                            data: (0..len).map(|_| self.next() as u8).collect(),
                        }
                    }),
                })
            }
            58 => S2CPlay::UpdateTileEntity(s2c::UpdateTileEntity {
                position: self.block_pos(),
                action: self.ubyte(),
                nbt: self.tile_nbt(),
            }),
            59 => S2CPlay::SignEditorOpen(s2c::SignEditorOpen {
                position: self.block_pos(),
            }),
            60 => S2CPlay::PlayerListItem(match self.below(5) {
                0 => s2c::PlayerListItem::AddPlayer(
                    (0..self.below(3))
                        .map(|_| s2c::AddPlayerEntry {
                            uuid: self.uuid(),
                            username: self.text(16),
                            properties: if self.flag() {
                                vec![s2c::ProfileProperty {
                                    name: "textures".to_owned(),
                                    value: self.text(40),
                                    signature: self.flag().then(|| self.text(40)),
                                }]
                            } else {
                                Vec::new()
                            },
                            game_mode: s2c::GameMode::Survival,
                            ping: self.int(),
                            display_name: self.flag().then(|| self.json()),
                        })
                        .collect(),
                ),
                1 => s2c::PlayerListItem::UpdateLatency(vec![s2c::UpdateLatencyEntry {
                    uuid: self.uuid(),
                    ping: self.int(),
                }]),
                2 => s2c::PlayerListItem::UpdateDisplayName(vec![s2c::UpdateDisplayNameEntry {
                    uuid: self.uuid(),
                    display_name: self.flag().then(|| self.json()),
                }]),
                3 => s2c::PlayerListItem::UpdateGameMode(vec![s2c::UpdateGameModeEntry {
                    uuid: self.uuid(),
                    game_mode: [
                        s2c::GameMode::NotSet,
                        s2c::GameMode::Survival,
                        s2c::GameMode::Creative,
                        s2c::GameMode::Adventure,
                        s2c::GameMode::Spectator,
                    ][self.below(5) as usize],
                }]),
                _ => s2c::PlayerListItem::RemovePlayer(vec![self.uuid()]),
            }),
            61 => S2CPlay::PlayerAbilities(s2c::PlayerAbilities {
                flags: self.ubyte(),
                fly_speed: self.motion_f32(),
                walk_speed: self.motion_f32(),
            }),
            62 => S2CPlay::TabComplete(s2c::TabComplete {
                matches: (0..self.below(3)).map(|_| self.text(20)).collect(),
            }),
            63 => S2CPlay::Title(match self.below(5) {
                0 => s2c::Title::SetTitle { text: self.json() },
                1 => s2c::Title::SetSubtitle { text: self.json() },
                2 => s2c::Title::SetTimes {
                    fade_in_ticks: self.int(),
                    stay_ticks: self.int(),
                    fade_out_ticks: self.int(),
                },
                3 => s2c::Title::Clear,
                _ => s2c::Title::Reset,
            }),
            64 => S2CPlay::WorldBorder(match self.below(6) {
                0 => s2c::WorldBorder::SetSize {
                    diameter: self.double(),
                },
                1 => s2c::WorldBorder::LerpSize {
                    old_diameter: self.double(),
                    new_diameter: self.double(),
                    speed_millis: self.long(),
                },
                2 => s2c::WorldBorder::SetCenter {
                    x: self.double(),
                    z: self.double(),
                },
                3 => s2c::WorldBorder::Initialize {
                    x: self.double(),
                    z: self.double(),
                    old_diameter: self.double(),
                    new_diameter: self.double(),
                    speed_millis: self.long(),
                    world_size: self.int(),
                    warning_blocks: self.int(),
                    warning_time: self.int(),
                },
                4 => s2c::WorldBorder::SetWarningTime {
                    warning_time: self.int(),
                },
                _ => s2c::WorldBorder::SetWarningBlocks {
                    warning_blocks: self.int(),
                },
            }),
            65 => S2CPlay::Camera(s2c::Camera {
                entity_id: self.entity_id(),
            }),
            66 => S2CPlay::CombatEvent(match self.below(3) {
                0 => s2c::CombatEvent::EnterCombat,
                1 => s2c::CombatEvent::EndCombat {
                    duration_ticks: self.int(),
                    entity_id: self.entity_id(),
                },
                _ => s2c::CombatEvent::EntityDead {
                    player_id: self.entity_id(),
                    entity_id: self.entity_id(),
                    death_message: self.text(50),
                },
            }),
            67 => S2CPlay::PluginMessage(s2c::PluginMessage {
                channel: [
                    "MC|Brand",
                    "MC|TrList",
                    "MC|BOpen",
                    "REGISTER",
                    "plugin:data",
                    "x",
                ][self.below(6) as usize]
                    .to_owned(),
                data: (0..self.below(40)).map(|_| self.next() as u8).collect(),
            }),
            68 => S2CPlay::UpdateEntityNBT(s2c::UpdateEntityNBT {
                entity_id: self.entity_id(),
                tag: self.nbt(),
            }),
            _ => S2CPlay::Statistics(s2c::Statistics {
                entries: (0..self.below(3))
                    .map(|_| s2c::StatisticEntry {
                        stat_id: ["stat.jump", "achievement.openInventory", "stat.mineBlock.1"]
                            [self.below(3) as usize]
                            .to_owned(),
                        value: self.int(),
                    })
                    .collect(),
            }),
        }
    }

    fn block_state_id(&mut self) -> i32 {
        match self.below(4) {
            0 => self.int(),
            1 => self.below(4096) as i32,
            _ => (self.below(198) as i32) << 4 | self.below(16) as i32,
        }
    }

    fn tile_nbt(&mut self) -> Option<NbtCompound> {
        if self.one_in(4) {
            return self.nbt();
        }
        let mut tag = NbtCompound::new();
        match self.below(6) {
            0 => {
                tag.insert("SpawnPotentials", Nbt::list(Vec::new()));
                tag.insert("EntityId", Nbt::String(self.text(16)));
                tag.insert("Delay", Nbt::Short(self.short()));
            }
            1 => {
                tag.insert("SkullType", Nbt::Byte(self.byte()));
                tag.insert("Rot", Nbt::Byte(self.byte()));
                tag.insert("ExtraType", Nbt::String(self.text(16)));
            }
            2 => {
                tag.insert("Levels", Nbt::Int(self.int()));
                tag.insert("Primary", Nbt::Int(self.int()));
                tag.insert("Secondary", Nbt::Int(self.int()));
            }
            3 => {
                tag.insert("Base", Nbt::Int(self.int()));
                let mut pattern = NbtCompound::new();
                pattern.insert("Pattern", Nbt::String(self.text(4)));
                pattern.insert("Color", Nbt::Int(self.int()));
                tag.insert("Patterns", Nbt::list(vec![Nbt::Compound(pattern)]));
            }
            4 => {
                tag.insert("Command", Nbt::String(self.text(40)));
                tag.insert("SuccessCount", Nbt::Int(self.int()));
                tag.insert("LastOutput", Nbt::String(self.json()));
                tag.insert("TrackOutput", Nbt::Byte(self.byte()));
            }
            _ => {
                tag.insert("Item", Nbt::String(self.text(20)));
                tag.insert("Data", Nbt::Int(self.int()));
            }
        }
        Some(tag)
    }

    /// The next packet the real decoder could have produced, after [`wire_reachable`].
    pub(crate) fn packet(&mut self) -> S2CPlay {
        loop {
            if let Some(packet) = wire_reachable(self.candidate()) {
                return packet;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sync::{apply, ClientState};

    fn join(entity_id: i32) -> S2CPlay {
        S2CPlay::JoinGame(s2c::JoinGame {
            entity_id,
            hardcore_mode: false,
            game_mode: 0,
            dimension: 0,
            difficulty: 2,
            max_players: 20,
            level_type: "default".into(),
            reduced_debug_info: false,
        })
    }

    /// Every wire-reachable packet through `sync::apply`, never a panic. An
    /// `Err` is fine: it is the logged handler exception vanilla's `Util.runTask` swallows.
    #[test]
    fn random_wire_reachable_packets_never_panic_the_dispatch() {
        const LOCAL: i32 = 1_000;
        let mut failures = Vec::new();
        for seed in 1..=6u64 {
            let mut fuzzer = PacketFuzzer::new(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15), LOCAL);
            let mut state = ClientState::new();
            let mut events = Vec::new();
            let _ = apply(&mut state, join(LOCAL), &mut events);
            for step in 0..4_000 {
                let packet = fuzzer.packet();
                let shown = format!("{packet:?}");
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let mut events = Vec::new();
                    let _ = apply(&mut state, packet, &mut events);
                }));
                if let Err(panic) = result {
                    let message = panic
                        .downcast_ref::<String>()
                        .cloned()
                        .or_else(|| panic.downcast_ref::<&str>().map(|s| (*s).to_owned()))
                        .unwrap_or_default();
                    failures.push(format!(
                        "seed {seed} step {step}: {message}\n  {shown:.600}"
                    ));
                    // A panic can leave the state half-updated; restart from a fresh join.
                    state = ClientState::new();
                    let _ = apply(&mut state, join(LOCAL), &mut Vec::new());
                }
            }
        }
        assert!(
            failures.is_empty(),
            "{} dispatch panics:\n{}",
            failures.len(),
            failures.join("\n")
        );
    }
}
