//! A timing probe, not a check. Run with
//! `cargo test -p mc-world --test nw_f5_biome_probe -- --ignored --nocapture`.
//!
//! On the frame a world is joined no column is loaded yet, so the sky and fog colour's biome
//! temperature (java `World.getFloatTemperature`) falls back to the seed-0 GenLayer provider,
//! whose first lookup in a 16x16 area generates the whole `BiomeCache` block. This prints what
//! that costs, against a repeat in the same block and a first lookup in another block.

use std::time::Instant;

use mc_math::BlockPos;
use mc_world::World;

#[test]
#[ignore = "timing probe; prints numbers"]
fn probe_first_biome_lookup_on_an_empty_world() {
    for round in 0..3 {
        let start = Instant::now();
        let world = World::new();
        let new_ms = start.elapsed().as_secs_f64() * 1000.0;
        // The SkyWars bench's spawn feet.
        let feet = BlockPos::new(-3892, 74, 2059);
        let start = Instant::now();
        let first = world.biome_at(feet);
        let first_ms = start.elapsed().as_secs_f64() * 1000.0;
        let start = Instant::now();
        let again = world.biome_at(feet);
        let again_ms = start.elapsed().as_secs_f64() * 1000.0;
        let start = Instant::now();
        let _ = world.biome_at(BlockPos::new(feet.x + 64, feet.y, feet.z + 64));
        let other_ms = start.elapsed().as_secs_f64() * 1000.0;
        assert_eq!(first, again);
        println!(
            "round {round}: World::new {new_ms:.3} ms, first lookup {first_ms:.3} ms, same block \
             {again_ms:.4} ms, another block {other_ms:.3} ms (biome {first})"
        );
    }
}
