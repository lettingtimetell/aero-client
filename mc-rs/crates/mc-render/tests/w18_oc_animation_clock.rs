//! The benchmark animation clock
//! (`mc_render::entity::item::pin_animation_clock`).
//!
//! Its own test binary, so pinning the process-wide clock cannot leak into other tests.
//!
//! * Unpinned (what a player runs: nothing in the client pins it) every reader returns the live
//!   clock it was given or the wall clock.
//! * Pinned, the glint's wall clock and every HUD clock hook read the pinned value; a glint clock
//!   pinned by a replay still wins for the glint; unpinning restores the live clocks.

use mc_render::entity::item::{
    animation_clock_ms, item_glint_system_time_ms, item_glint_wall_clock_ms, pin_animation_clock,
    pin_item_glint_clock,
};

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

#[test]
fn the_animation_clock_is_live_unless_a_benchmark_pins_it() {
    // Unpinned: the live values pass straight through.
    assert_eq!(animation_clock_ms(12_345), 12_345);
    let before = now_ms();
    let glint = item_glint_wall_clock_ms();
    let after = now_ms();
    assert!(
        (before..=after).contains(&glint),
        "the glint follows the wall clock: {before} <= {glint} <= {after}"
    );
    assert!((before..=now_ms()).contains(&item_glint_system_time_ms()));

    // Pinned: every reader sees the pinned value, whatever the live one.
    pin_animation_clock(Some(4_000));
    assert_eq!(animation_clock_ms(12_345), 4_000);
    assert_eq!(animation_clock_ms(0), 4_000);
    assert_eq!(item_glint_wall_clock_ms(), 4_000);
    assert_eq!(item_glint_system_time_ms(), 4_000);

    // A replay's glint pin wins for the glint; the HUD clocks keep the benchmark's.
    pin_item_glint_clock(Some(77));
    assert_eq!(item_glint_system_time_ms(), 77);
    assert_eq!(animation_clock_ms(12_345), 4_000);
    pin_item_glint_clock(None);
    assert_eq!(item_glint_system_time_ms(), 4_000);

    // Unpinned again: live.
    pin_animation_clock(None);
    assert_eq!(animation_clock_ms(12_345), 12_345);
    let before = now_ms();
    let glint = item_glint_system_time_ms();
    assert!((before..=now_ms()).contains(&glint));
}
