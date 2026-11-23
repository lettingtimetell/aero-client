//! Bounded, owner-held CPU geometry capacity. No GPU resource or game state is retained.
use mc_render::entity::EntityVertex;

pub(crate) const MAX_BUFFER_BYTES: usize = 1024 * 1024;

#[derive(Default)]
pub(crate) struct EntityScratch {
    pub vertices: Vec<EntityVertex>,
    pub layer_vertices: Vec<EntityVertex>,
    pub layer_body: Vec<EntityVertex>,
}

pub(crate) fn recycle<T>(slot: &mut Vec<T>, mut values: Vec<T>) {
    values.clear();
    if values.capacity().saturating_mul(std::mem::size_of::<T>()) > MAX_BUFFER_BYTES {
        values = Vec::new();
    }
    *slot = values;
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reuse_clears_live_data_and_large_bursts_do_not_pin_memory() {
        let values = vec![42u32; 1024];
        let address = values.as_ptr();
        let mut slot = Vec::new();
        recycle(&mut slot, values);
        assert!(slot.is_empty());
        assert_eq!(slot.as_ptr(), address);
        recycle(&mut slot, Vec::with_capacity(MAX_BUFFER_BYTES / 4 + 1));
        assert_eq!(slot.capacity(), 0);
    }
}
