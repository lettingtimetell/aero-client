//! Protocol-5 (1.7.10) block cells decoded to the client's 1.8.9 block states.
//!
//! A 1.7.10 cell is `id << 4 | meta` with a 12-bit id: the same packing as a 1.8.9 state, but
//! 1.7.10 resolves the block by id alone and lets each block class read the meta, while 1.8.9
//! only accepts the values its state table registers and reads every other value as air
//! (java17: `ExtendedBlockStorage.getBlockByExtId` / `getExtBlockMetadata`; java18:
//! `ExtendedBlockStorage.get`). The client keeps 1.8.9 storage, so every 1.7.10 cell is
//! rewritten once, where the packet is decoded, to a registered 1.8.9 state with the same
//! behaviour: chunk sections, block changes, falling blocks and carried blocks. Nothing here is
//! reachable from a 1.8.9 session.
//!
//! The table and the rule behind each entry are in `extractor/codegen_v5.py`. In short:
//! unregistered 1.7.10 ids are air; a meta 1.8.9 registers is kept, so the stored state still
//! carries the raw 1.7.10 meta; any other meta takes 1.8.9's `getStateFromMeta` choice, except
//! for the documented families where the 1.7.10 block code does something else with it.

use crate::block::BlockState;

use super::generated::V5_STATE_REMAP;

/// The 1.8.9 state a protocol-5 cell `id << 4 | meta` is stored as.
///
/// Total over `u16`: a 12-bit id from 256 up (the chunk's add nibble set) is never registered
/// in 1.7.10 and decodes to air, as 1.7.10's defaulted block registry does.
///
/// # Example
/// ```
/// use mc_data::v5::state_from_v5;
/// use mc_data::{BlockId, BlockState};
///
/// // A torch with meta 0 is a standing torch in 1.7.10; 1.8.9 has no meta-0 torch.
/// assert_eq!(state_from_v5(50 << 4), BlockState::new(BlockId(50), 5));
/// // Slime (165) is a 1.8.9 block; 1.7.10 has nothing at that id.
/// assert_eq!(state_from_v5(165 << 4), BlockState::AIR);
/// ```
#[inline]
#[must_use]
pub fn state_from_v5(raw: u16) -> BlockState {
    match V5_STATE_REMAP.get(usize::from(raw)) {
        Some(&state) => BlockState::from_raw(state),
        None => BlockState::AIR,
    }
}

/// [`state_from_v5`] for a block id and meta carried separately, as `S23PacketBlockChange`
/// does (a VarInt id and an unsigned-byte meta).
///
/// An id outside `0..4096` is air, as `Block.getBlockById` returns the registry default; only
/// the low four bits of the meta are stored (java17: `Chunk.func_150807_a` writes the meta into
/// a nibble array).
#[inline]
#[must_use]
pub fn state_from_v5_id_meta(id: i32, meta: i32) -> BlockState {
    if (0..4096).contains(&id) {
        // `id < 4096` and the mask keep both halves in range, so the packing cannot overflow.
        state_from_v5(((id as u16) << 4) | (meta & 0x0F) as u16)
    } else {
        BlockState::AIR
    }
}

/// Whether 1.7.10 registers block `id`: 0-164 and 170-175 (java17: `Block.registerBlocks`).
#[inline]
#[must_use]
pub fn is_block_registered_v5(id: u16) -> bool {
    id == 0 || (id < 256 && state_from_v5(id << 4).block().0 == id)
}

/// The block of a protocol-5 falling block, from `S0EPacketSpawnObject` type 70's data int:
/// block id `data & 0xFFFF`, meta `data >> 16` (java17:
/// `NetHandlerPlayClient.handleSpawnObject`). 1.8.9 packs the same data as a
/// `Block.getStateId`, which this replaces.
#[inline]
#[must_use]
pub fn state_from_v5_falling_block(data: i32) -> BlockState {
    state_from_v5_id_meta(data & 0xFFFF, data >> 16)
}

/// The block an enderman carries in 1.7.10: its watched byte 16 is the block id and byte 17 the
/// meta, both read as signed bytes (java17: `EntityEnderman.getCarriedBlock` passes the byte to
/// `Block.getBlockById` without a mask, so a negative byte is air; `getCarryingData`). The
/// setters mask with `& 255`, and every block an enderman picks up has an id below 128.
#[inline]
#[must_use]
pub fn state_from_v5_enderman(carried_id: i8, carried_meta: i8) -> BlockState {
    state_from_v5_id_meta(i32::from(carried_id), i32::from(carried_meta))
}

/// Converts one protocol-5 chunk section's block arrays into the 1.8.9 section blob layout
/// (`Chunk.fillChunk`'s two bytes per block, little-endian, in index order `y << 8 | z << 4 |
/// x`), through [`state_from_v5`].
///
/// `lsb` is the section's block-id array, `meta` its metadata nibbles and `add` its optional
/// add (MSB) nibbles; nibble `i` is the low half of byte `i >> 1` when `i` is even (java17:
/// `NibbleArray.get`). A section without an add array has a zero high id nibble everywhere,
/// which is what a full chunk leaves behind (java17: `Chunk.fillChunk` clears the MSB array).
///
/// # Example
/// ```
/// use mc_data::v5::write_section_blob_v5;
///
/// let mut lsb = [0u8; 4096];
/// let mut meta = [0u8; 2048];
/// lsb[1] = 50; // a torch at x = 1
/// meta[0] = 0x00; // meta 0 for both block 0 and block 1
/// let mut out = [0u8; 8192];
/// write_section_blob_v5(&lsb, &meta, None, &mut out);
/// // Block 1 is stored as the standing torch, 1.8.9 state 50 << 4 | 5 = 805.
/// assert_eq!(u16::from_le_bytes([out[2], out[3]]), 805);
/// ```
pub fn write_section_blob_v5(
    lsb: &[u8; 4096],
    meta: &[u8; 2048],
    add: Option<&[u8; 2048]>,
    out: &mut [u8; 8192],
) {
    // Absent add nibbles read as zero.
    let add = add.unwrap_or(&[0; 2048]);
    // Each meta/add byte covers two consecutive blocks: four output bytes, two id bytes.
    for (((pair, ids), &meta_byte), &add_byte) in out
        .chunks_exact_mut(4)
        .zip(lsb.chunks_exact(2))
        .zip(meta)
        .zip(add)
    {
        let low = cell(ids[0], meta_byte & 0x0F, add_byte & 0x0F);
        let high = cell(ids[1], meta_byte >> 4, add_byte >> 4);
        pair[..2].copy_from_slice(&low.id.to_le_bytes());
        pair[2..].copy_from_slice(&high.id.to_le_bytes());
    }
}

#[inline]
fn cell(lsb: u8, meta: u8, add: u8) -> BlockState {
    state_from_v5((u16::from(add) << 12) | (u16::from(lsb) << 4) | u16::from(meta))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::BlockId;

    /// Ids 1.7.10 registers (java17: `Block.registerBlocks`, block/Block.java:171-381).
    fn registered(id: u16) -> bool {
        id <= 164 || (170..=175).contains(&id)
    }

    /// The explicit choices, written out a second time independently of the generator, as
    /// `(block, 1.7.10 meta, 1.8.9 meta)`.
    fn overridden(block: u16, meta: u8) -> Option<u8> {
        Some(match (block, meta) {
            (1 | 19, _) => 0,
            (6, 6 | 7) => 5,
            (6, 14 | 15) => 13,
            (27 | 28 | 157, 6 | 7) => 0,
            (27 | 28 | 157, 14 | 15) => 8,
            (29 | 33, 6 | 7 | 14 | 15) => 2,
            (34 | 36, 6 | 7) => 5,
            (34 | 36, 14 | 15) => 13,
            (50 | 75 | 76, 9..=12) => meta & 7,
            (59 | 141 | 142 | 104 | 105, 8..=15) => 7,
            (92, 7..=15) => 6,
            (115 | 118, 4..=15) => 3,
            (127, 12..=15) => 8 | (meta & 3),
            (145, 12..=15) => meta & 3,
            (154, 1 | 7) => 0,
            (154, 9 | 15) => 8,
            (161, _) if meta & 3 >= 2 => meta & 12,
            (162, _) if meta & 3 >= 2 => (meta & 12) | ((meta & 3) % 2),
            _ => return None,
        })
    }

    /// The whole table, recomputed from mc-data's own 1.8.9 tables: the legality of every
    /// packed state (`BLOCK_STATE_IDS`) and the virtual `getStateFromMeta` image
    /// (`try_from_java_state_id`), plus the explicit choices above.
    #[test]
    fn table_matches_the_rules_over_mc_datas_own_1_8_9_tables() {
        let mut overrides = 0;
        let mut canonicalised = 0;
        for raw in 0..=u16::MAX {
            let (block, meta) = (raw >> 4, (raw & 0x0F) as u8);
            let got = state_from_v5(raw);
            let expected = if !registered(block) {
                BlockState::AIR
            } else if let Some(target) = overridden(block, meta) {
                overrides += 1;
                BlockState::new(BlockId(block), target)
            } else if BlockState::new(BlockId(block), meta).is_legal() {
                BlockState::new(BlockId(block), meta)
            } else {
                canonicalised += 1;
                BlockState::try_from_java_state_id(i32::from(block) | (i32::from(meta) << 12))
                    .expect("a throwing meta must have an explicit choice")
            };
            assert_eq!(got, expected, "1.7.10 block {block} meta {meta}");
        }
        // Negative control on the counts: a table that skipped the overrides or the
        // canonicalisation would not reach these.
        assert_eq!(overrides, 177);
        assert_eq!(canonicalised, 1392);
    }

    #[test]
    fn every_registered_cell_is_a_registered_state_of_the_same_block() {
        for raw in 0..=u16::MAX {
            let state = state_from_v5(raw);
            let block = raw >> 4;
            assert!(state.is_legal(), "{raw:#06x} -> {:#06x}", state.id);
            if registered(block) {
                assert_eq!(state.block().0, block, "{raw:#06x}");
                // The 1.8.9 paths that decode `Block.getStateId` (falling blocks, carried
                // blocks, particles) must never meet a throwing meta.
                assert!(BlockState::try_from_java_state_id(state.to_java_state_id()).is_some());
            } else {
                assert_eq!(state, BlockState::AIR, "{raw:#06x}");
            }
        }
    }

    #[test]
    fn registration_matches_1_7_10() {
        for id in 0..4096u16 {
            assert_eq!(is_block_registered_v5(id), registered(id), "id {id}");
        }
        // 1.8.9 registers these, 1.7.10 does not.
        for id in [165, 166, 167, 168, 169, 176, 197] {
            assert!(BlockId(id).is_registered() && !is_block_registered_v5(id));
        }
    }

    #[test]
    fn hand_checked_cells() {
        let s = |b, m| BlockState::new(BlockId(b), m);
        // `(1.7.10 block, meta, decoded state, whether 1.8.9 registers the raw value)`. A raw
        // value 1.8.9 does not register reads as air in a 1.8.9 lookup, which is the failure
        // the table exists to avoid; a registered one is kept unless 1.7.10 reads it otherwise.
        for (raw_block, raw_meta, want, raw_legal) in [
            (4, 1, s(4, 0), false),      // cobblestone 1
            (54, 0, s(54, 2), false),    // chest 0 -> north
            (61, 0, s(61, 2), false),    // furnace 0 -> north
            (50, 0, s(50, 5), false),    // torch 0 -> standing
            (53, 9, s(53, 1), false),    // stairs 9 -> 1
            (64, 12, s(64, 8), false),   // door upper 12 -> 8
            (50, 10, s(50, 2), false),   // torch 10 traces as a west-facing wall torch
            (34, 7, s(34, 5), false),    // piston head 7 is facing 5 in 1.7.10
            (59, 12, s(59, 7), false),   // wheat 12 draws ripe
            (145, 14, s(145, 2), false), // anvil 14: undamaged, facing 2
            (1, 3, s(1, 0), true),       // stone 3 is plain stone, not 1.8.9's diorite
            (19, 1, s(19, 0), true),     // sponge 1 is dry, not 1.8.9's wet sponge
            (107, 8, s(107, 8), true),   // fence gate 8 keeps its raw meta
            (27, 11, s(27, 11), true),   // powered ascending rail keeps its raw meta
        ] {
            let raw = BlockState::new(BlockId(raw_block), raw_meta as u8);
            assert_eq!(state_from_v5(raw.id), want, "{raw_block}:{raw_meta}");
            assert_eq!(raw.is_legal(), raw_legal, "{raw_block}:{raw_meta}");
            if !raw_legal {
                assert!(
                    raw.props().is_air(),
                    "the raw decode loses {raw_block}:{raw_meta}"
                );
            }
        }
    }

    #[test]
    fn separate_id_and_meta_forms() {
        assert_eq!(state_from_v5_id_meta(50, 0), state_from_v5(50 << 4));
        // Only the meta nibble is stored; the S23 meta byte can be up to 255.
        assert_eq!(
            state_from_v5_id_meta(35, 0x1E),
            state_from_v5((35 << 4) | 0x0E)
        );
        assert_eq!(state_from_v5_id_meta(4096 + 1, 0), BlockState::AIR);
        assert_eq!(state_from_v5_id_meta(-1, 0), BlockState::AIR);
        // Falling block: id in the low half, meta in the high half.
        assert_eq!(
            state_from_v5_falling_block(12 | (1 << 16)),
            state_from_v5((12 << 4) | 1)
        );
        assert_eq!(
            state_from_v5_falling_block(0x1_0000 | 0xFFFF),
            BlockState::AIR
        );
        // 1.8.9's packing (`id | meta << 12`) must not be what this decodes.
        assert_ne!(
            state_from_v5_falling_block(12 | (1 << 12)),
            state_from_v5((12 << 4) | 1)
        );
        // Enderman: `getCarriedBlock` passes the signed byte to `getBlockById` unmasked, so a
        // negative byte is air, not block `byte & 255` (-128 is not sandstone stairs).
        assert_eq!(state_from_v5_enderman(-0x80, 0), BlockState::AIR);
        assert_eq!(state_from_v5_enderman(-1, 3), BlockState::AIR);
        assert_ne!(
            state_from_v5(128 << 4),
            BlockState::AIR,
            "128 is a real block"
        );
        assert_eq!(state_from_v5_enderman(127, 0), state_from_v5(127 << 4));
        assert_eq!(state_from_v5_enderman(2, 0), state_from_v5(2 << 4));
        assert_eq!(state_from_v5_enderman(98, 3), state_from_v5((98 << 4) | 3));
        // The meta byte is read unmasked too; only its low nibble is a storage meta.
        assert_eq!(
            state_from_v5_enderman(98, -13),
            state_from_v5((98 << 4) | 3)
        );
    }

    #[test]
    fn section_blob_layout() {
        let mut lsb = [0u8; 4096];
        let mut meta = [0u8; 2048];
        let mut add = [0u8; 2048];
        // Block 0: torch meta 3 (low nibble). Block 1: stone meta 2 (high nibble).
        lsb[0] = 50;
        lsb[1] = 1;
        meta[0] = 0x23;
        // Last block: id 0x101 via the add nibble (high half of the last byte).
        lsb[4095] = 1;
        add[2047] = 0x10;
        fn at(out: &[u8; 8192], i: usize) -> u16 {
            u16::from_le_bytes([out[2 * i], out[2 * i + 1]])
        }
        let mut out = [0xAAu8; 8192];
        write_section_blob_v5(&lsb, &meta, Some(&add), &mut out);
        assert_eq!(at(&out, 0), (50 << 4) | 3);
        assert_eq!(at(&out, 1), 1 << 4);
        assert_eq!(at(&out, 2), 0);
        assert_eq!(at(&out, 4095), 0, "id 257 is unregistered in 1.7.10");
        // Without the add array the same cell is stone.
        write_section_blob_v5(&lsb, &meta, None, &mut out);
        assert_eq!(at(&out, 4095), 1 << 4);
    }
}
