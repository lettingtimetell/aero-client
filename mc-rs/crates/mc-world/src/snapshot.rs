//! Exact copies of a client world, for replay checkpoints.
//!
//! A replay seek restores a checkpoint and plays forward from it, and the frame it reaches must
//! be byte-identical to the one reached by playing from the start. So a copy here is the whole
//! world, not a render view of it ([`World::render_snapshot`] is that): every chunk, tile entity,
//! map, light-work cursor and random stream, and every entity in the `hecs` arena with the same
//! [`Entity`] handle it has now.
//!
//! Chunk columns are `Arc`s that every writer changes through `Arc::make_mut`, so a copy shares
//! each column with the live world until one of them writes it: a copy costs one reference count
//! per column plus the entity arena, and its memory is what changed since.
//!
//! The arena holds components of types this crate does not know (the client attaches its own),
//! so a copy is made through an [`EcsCloner`] that lists every component type it can copy. A world
//! holding a component of any other type is **not** copied ([`World::try_clone_with`] returns
//! `None`): a component nobody registered can never be dropped from a copy silently.
//!
//! Entity handles: each entity is re-created with `hecs::World::spawn_at` under its own handle,
//! in the arena's iteration order (archetype by archetype, in storage order), so every handle the
//! client keeps (the entity tracker, riding and leash links) stays valid. The arena's free-list
//! order is not copied (hecs does not expose it), so an entity spawned after a restore may get a
//! different handle than it would have; nothing in the client orders or hashes by the handle's
//! numeric value (entity walks follow the tracker's `iteration_order`), so that is unobservable.

use std::any::TypeId;

use hecs::{EntityBuilder, EntityRef};

use crate::world::World;

/// Copies one component of a known type from an entity into a builder.
type CloneFn = fn(&EntityRef<'_>, &mut EntityBuilder);

/// The component types a world copy knows how to clone (see the module docs).
#[derive(Debug, Clone, Default)]
pub struct EcsCloner {
    entries: Vec<(TypeId, CloneFn)>,
}

fn clone_component<T: hecs::Component + Clone>(entity: &EntityRef<'_>, out: &mut EntityBuilder) {
    if let Some(component) = entity.get::<&T>() {
        out.add((*component).clone());
    }
}

impl EcsCloner {
    /// A cloner that knows this crate's own components ([`crate::entity`]). The client adds its
    /// own with [`Self::with`].
    #[must_use]
    pub fn with_world_components() -> Self {
        Self::default()
            .with::<crate::entity::Position>()
            .with::<crate::entity::Velocity>()
            .with::<crate::entity::Rotation>()
            .with::<crate::entity::EntityKind>()
            .with::<crate::entity::Metadata>()
            .with::<crate::entity::Acceleration>()
            .with::<crate::entity::ItemFrameFacing>()
            .with::<crate::entity::SplashPotionDamage>()
            .with::<crate::entity::ArrowShooter>()
    }

    /// Also copy components of type `T`.
    #[must_use]
    pub fn with<T: hecs::Component + Clone>(mut self) -> Self {
        let id = TypeId::of::<T>();
        if !self.entries.iter().any(|(known, _)| *known == id) {
            self.entries.push((id, clone_component::<T>));
        }
        self
    }

    /// The number of component types this cloner knows.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether it knows none.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    fn find(&self, id: TypeId) -> Option<CloneFn> {
        self.entries
            .iter()
            .find(|(known, _)| *known == id)
            .map(|(_, clone)| *clone)
    }
}

impl World {
    /// An exact, independent copy of this world, or `None` when the entity arena holds a
    /// component type `cloner` does not know (see the module docs).
    #[must_use]
    pub fn try_clone_with(&self, cloner: &EcsCloner) -> Option<World> {
        let mut ecs = hecs::World::new();
        for entity in self.ecs.iter() {
            let mut builder = EntityBuilder::new();
            for id in entity.component_types() {
                let clone = cloner.find(id)?;
                clone(&entity, &mut builder);
            }
            ecs.spawn_at(entity.entity(), builder.build());
        }
        Some(World {
            chunks: self.chunks.clone(),
            blank_chunk: self.blank_chunk.clone(),
            ecs,
            dirty: self.dirty.clone(),
            has_sky_light: self.has_sky_light,
            biome_source: self.biome_source,
            level_type: self.level_type,
            debug_world_states: self.debug_world_states.clone(),
            debug_world_grid_width: self.debug_world_grid_width,
            debug_world_grid_height: self.debug_world_grid_height,
            overworld_biome_provider: self.overworld_biome_provider.clone(),
            tile_entities: self.tile_entities.clone(),
            light_update_block_list: Vec::new(),
            light_tick_queue: self.light_tick_queue.clone(),
            previous_active_light_chunks: self.previous_active_light_chunks.clone(),
            active_light_chunks: self.active_light_chunks.clone(),
            reference_light_engine: self.reference_light_engine,
            map_data: self.map_data.clone(),
            map_data_metadata: self.map_data_metadata.clone(),
            tile_entity_random: self.tile_entity_random.clone(),
        })
    }

    /// Mark every resident column (and its neighbours' borders) for meshing, exactly as loading
    /// each column does: what a renderer needs after a restored copy replaced the world it had
    /// meshed. Walks the columns sorted, so the dirty set never depends on the map's hashing.
    pub fn mark_every_column_dirty(&mut self) {
        let mut columns: Vec<mc_math::ChunkPos> = self.chunks.keys().copied().collect();
        columns.sort_unstable_by_key(|pos| (pos.x, pos.z));
        for pos in columns {
            self.mark_column_and_neighbours_dirty(pos);
        }
    }

    /// The number of resident chunk columns.
    #[must_use]
    pub fn column_count(&self) -> usize {
        self.chunks.len()
    }

    /// About how many bytes this copy holds that `other` (the copy before it, if any) does not
    /// share: every chunk column and section that is not the same `Arc`, plus the tile entities
    /// and the entity arena it copied. An estimate for a memory budget (heap payloads such as
    /// sign text and NBT are not walked).
    #[must_use]
    pub fn bytes_not_shared_with(&self, other: Option<&World>) -> usize {
        let section = std::mem::size_of::<crate::chunk::ChunkSection>();
        let column = std::mem::size_of::<crate::chunk::Chunk>();
        let mut bytes = 0usize;
        for (pos, chunk) in &self.chunks {
            let theirs = other.and_then(|other| other.chunks.get(pos));
            if theirs.is_some_and(|theirs| std::sync::Arc::ptr_eq(chunk, theirs)) {
                continue;
            }
            bytes += column;
            for (index, mine) in chunk.sections().iter().enumerate() {
                let Some(mine) = mine else {
                    continue;
                };
                let shared = theirs
                    .and_then(|theirs| theirs.sections()[index].as_ref())
                    .is_some_and(|theirs| std::sync::Arc::ptr_eq(mine, theirs));
                if !shared {
                    bytes += section;
                }
            }
        }
        bytes += self.tile_entities.len()
            * std::mem::size_of::<(mc_math::BlockPos, crate::tileent::TileEntity)>();
        bytes += self.ecs.len() as usize * 256;
        bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entity::{EntityBundle, EntityKind, Metadata, Position, Rotation, Velocity};
    use mc_math::DVec3;

    #[derive(Debug, Clone, PartialEq)]
    struct Unknown(u8);

    fn bundle(x: f64) -> EntityBundle {
        EntityBundle {
            position: Position(DVec3::new(x, 64.0, 0.0)),
            velocity: Velocity(DVec3::ZERO),
            rotation: Rotation::default(),
            kind: EntityKind::Player,
            metadata: Metadata::new(),
        }
    }

    #[test]
    fn a_copy_keeps_every_entity_handle_and_component_and_is_independent() {
        let mut world = World::new();
        let a = world.spawn(bundle(1.0));
        let b = world.spawn(bundle(2.0));
        let c = world.spawn(bundle(3.0));
        world.despawn(b).expect("b exists");
        let _ = world
            .ecs
            .insert_one(c, crate::entity::Acceleration(DVec3::new(0.0, 1.0, 0.0)));
        let cloner = EcsCloner::with_world_components();
        let copy = world
            .try_clone_with(&cloner)
            .expect("every component is known");
        assert!(copy.ecs.contains(a) && copy.ecs.contains(c) && !copy.ecs.contains(b));
        assert_eq!(copy.ecs.len(), 2);
        let pos = |w: &World, e| w.ecs.get::<&Position>(e).map(|p| p.0.x).ok();
        assert_eq!(pos(&copy, a), Some(1.0));
        assert_eq!(pos(&copy, c), Some(3.0));
        assert!(copy.ecs.get::<&crate::entity::Acceleration>(c).is_ok());
        // Independent: moving the copy's entity leaves the original alone.
        copy.ecs.get::<&mut Position>(a).expect("a").0.x = 9.0;
        assert_eq!(pos(&world, a), Some(1.0));
        // The same walk order.
        let order = |w: &World| w.ecs.iter().map(|e| e.entity()).collect::<Vec<_>>();
        assert_eq!(order(&world), order(&copy));
        // A copy of an empty-terrain world holds only its two entities' estimate.
        assert_eq!(copy.bytes_not_shared_with(Some(&world)), 2 * 256);
        assert_eq!(copy.bytes_not_shared_with(None), 2 * 256);
    }

    #[test]
    fn a_copy_shares_chunk_sections_until_one_side_writes() {
        use mc_data::{BlockId, BlockState};
        use mc_math::{BlockPos, ChunkPos};
        let stone = BlockState::new(BlockId(1), 0);
        let mut live = World::new();
        live.load_empty_chunk(ChunkPos::new(0, 0));
        live.load_empty_chunk(ChunkPos::new(1, 0));
        live.set_block(BlockPos::new(1, 20, 1), stone);
        live.set_block(BlockPos::new(17, 20, 1), stone);
        let cloner = EcsCloner::with_world_components();
        let first = live.try_clone_with(&cloner).expect("copy");
        let section = std::mem::size_of::<crate::chunk::ChunkSection>();
        let column = std::mem::size_of::<crate::chunk::Chunk>();
        // Alone, the copy accounts for both columns and their one section each.
        assert_eq!(first.bytes_not_shared_with(None), 2 * (column + section));
        // A second copy right away shares everything with the first.
        let same = live.try_clone_with(&cloner).expect("copy");
        assert_eq!(same.bytes_not_shared_with(Some(&first)), 0);
        // The live world writes one block: a copy taken now holds that column and section only.
        live.set_block(BlockPos::new(1, 21, 1), stone);
        let later = live.try_clone_with(&cloner).expect("copy");
        assert_eq!(later.bytes_not_shared_with(Some(&first)), column + section);
        // The first copy still has the old block state: copies are independent.
        assert_eq!(first.block_at(BlockPos::new(1, 21, 1)), BlockState::AIR);
        assert_eq!(later.block_at(BlockPos::new(1, 21, 1)), stone);
    }

    #[test]
    fn a_world_with_an_unregistered_component_is_not_copied() {
        let mut world = World::new();
        let a = world.spawn(bundle(1.0));
        let _ = world.ecs.insert_one(a, Unknown(7));
        assert!(world
            .try_clone_with(&EcsCloner::with_world_components())
            .is_none());
        let cloner = EcsCloner::with_world_components().with::<Unknown>();
        let copy = world.try_clone_with(&cloner).expect("now known");
        assert_eq!(copy.ecs.get::<&Unknown>(a).map(|u| u.0).ok(), Some(7));
    }
}
