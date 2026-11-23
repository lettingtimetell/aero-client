//! `mc_render::frame::Timer` is vanilla's tick clock.
//!
//! A server-side anticheat counts movement packets against wall time, and the client sends one per
//! tick, so the tick clock decides whether the client looks like a timer cheat (more than 20 a
//! second) or lags (fewer). Two kinds of evidence:
//!
//! * **Frame-for-frame identity with the real class.** `fixtures/timer_ref.bin` is written by
//!   `javaref/TimerRef.java` (`javaref/generate-timer.ps1`), which runs the recompiled 1.8.9
//!   `net/minecraft/util/Timer.class` with only its two clock reads redirected. Its frames cover
//!   360 Hz, random frame times, hitches either side of the 10-tick catch-up cap (500 ms) and of
//!   the 1000 ms clock-sync window, two drifting clocks and sub-millisecond frames. Every frame's
//!   `elapsedTicks` and the exact bits of `renderPartialTicks` must match.
//! * **The live feed runs exactly 20 ticks a second.** The shell feeds one millisecond clock to
//!   both of the timer's inputs (`Timer::update`); ten minutes at any frame rate must come out at
//!   12 000 ticks, with no frame over the cap.

use mc_render::frame::{Timer, MAX_CATCH_UP_TICKS, TICKS_PER_SECOND};

const FRAME_BYTES: usize = 13;

/// `(system-clock delta, high-resolution-clock delta, elapsedTicks, renderPartialTicks bits)`.
fn reference_frames() -> Vec<(i64, i64, i32, u32)> {
    let bytes: &[u8] = include_bytes!("fixtures/timer_ref.bin");
    assert_eq!(&bytes[..8], b"L1TIMER1", "fixture magic");
    let count = u32::from_be_bytes(bytes[8..12].try_into().expect("count")) as usize;
    let body = &bytes[12..];
    assert_eq!(body.len(), count * FRAME_BYTES, "fixture length");
    body.chunks_exact(FRAME_BYTES)
        .map(|frame| {
            let word = |at: usize| u32::from_be_bytes(frame[at..at + 4].try_into().expect("u32"));
            (
                i64::from(word(0)),
                i64::from(word(4)),
                i32::from(frame[8]),
                word(9),
            )
        })
        .collect()
}

#[test]
fn timer_matches_the_vanilla_timer_class_frame_for_frame() {
    let frames = reference_frames();
    assert_eq!(frames.len(), 11_010, "the generator's frame count");
    // `Timer::new` is the state `new Timer(20.0F)` has with both clocks reading 0, and the state
    // the live shell starts from (`reset_clocks(0, 0)` at window creation).
    let mut timer = Timer::new(TICKS_PER_SECOND);
    let (mut sys, mut hr) = (0_i64, 0_i64);
    let mut ticks = 0_i64;
    let mut capped = 0;
    for (index, &(sys_delta, hr_delta, java_ticks, java_partial)) in frames.iter().enumerate() {
        sys += sys_delta;
        hr += hr_delta;
        timer.update_with(hr, sys);
        assert_eq!(
            (
                timer.elapsed_ticks(),
                timer.render_partial_ticks().to_bits()
            ),
            (java_ticks, java_partial),
            "frame {index}: sys {sys} ms, hr {hr} ms (java partial {})",
            f32::from_bits(java_partial)
        );
        ticks += i64::from(java_ticks);
        if java_ticks == MAX_CATCH_UP_TICKS {
            capped += 1;
        }
    }
    // The trace really exercises the cap: the 500, 501, 700, 999 and 1000 ms hitches reach it
    // (`j <= 1000` keeps the 1000 ms one inside the sync window), while the 1001, 1500 and
    // 2500 ms ones fall outside it and run no ticks at all.
    assert!(capped >= 5, "the cap was reached {capped} times");
    assert!(ticks > 0);

    // Control: the trace tells the two clocks apart. Swapping them changes when
    // `timeSyncAdjustment` moves (the drifting segment), so frames stop matching.
    let mut swapped = Timer::new(TICKS_PER_SECOND);
    let (mut sys, mut hr) = (0_i64, 0_i64);
    let mut mismatches = 0;
    for &(sys_delta, hr_delta, java_ticks, java_partial) in &frames {
        sys += sys_delta;
        hr += hr_delta;
        swapped.update_with(sys, hr);
        if (
            swapped.elapsed_ticks(),
            swapped.render_partial_ticks().to_bits(),
        ) != (java_ticks, java_partial)
        {
            mismatches += 1;
        }
    }
    assert!(
        mismatches > 100,
        "swapped clocks still matched ({mismatches} mismatches)"
    );
}

/// java: `updateTimer`'s `if (j <= 1000L && j >= 0L) ... else lastHRTime = d0`. A frame that
/// took more than a second runs **no** ticks (its time is discarded), and because the
/// high-resolution clock kept running while `counter` did not, the next clock sync pulls
/// `timeSyncAdjustment` below 1: the following seconds tick slower than 20 a second. The Rust
/// timer reproduces that, quirk included; this pins the numbers the parity test proves.
#[test]
fn a_stall_over_one_second_drops_its_time_and_briefly_slows_the_clock_like_vanilla() {
    let mut timer = Timer::new(TICKS_PER_SECOND);
    let mut clock = 0_i64;
    let mut frame = |timer: &mut Timer, millis: i64| {
        clock += millis;
        timer.update_with(clock, clock);
        timer.elapsed_ticks()
    };
    // Settle: 5 s at 10 ms.
    for _ in 0..500 {
        frame(&mut timer, 10);
    }
    assert_eq!(
        frame(&mut timer, 1500),
        0,
        "the stalled frame's time is dropped"
    );
    // The next five seconds of 10 ms frames.
    let after: i32 = (0..500).map(|_| frame(&mut timer, 10)).sum();
    assert!(
        (80..100).contains(&after),
        "vanilla's clock-sync quirk runs the next 5 s slow, got {after} ticks"
    );
}

/// The live shell (`Timer::update`) feeds one monotonic millisecond clock to both inputs, so
/// `timeSyncAdjustment` stays exactly 1 and the long-run rate is exactly 20 ticks a second at
/// any frame rate. A client that ran even 1 % fast would send 12 extra movement packets a minute.
#[test]
fn the_live_clock_feed_runs_exactly_twenty_ticks_a_second_at_any_frame_rate() {
    const TEN_MINUTES_US: i64 = 600_000_000;
    // Frame periods in microseconds: 2600 fps, 1 ms, 360/240/144/60/30/20 Hz, and frame times
    // that straddle a tick (49/51 ms) or need two (99 ms) or several (333 ms) per frame.
    for period in [
        385_i64, 1_000, 2_778, 4_167, 6_944, 16_667, 33_333, 49_000, 50_000, 51_000, 99_000,
        333_000,
    ] {
        let mut timer = Timer::new(TICKS_PER_SECOND);
        let mut micros = 0_i64;
        let mut ticks = 0_i64;
        let mut most = 0;
        while micros + period <= TEN_MINUTES_US {
            micros += period;
            let millis = micros / 1_000;
            timer.update_with(millis, millis);
            most = most.max(timer.elapsed_ticks());
            ticks += i64::from(timer.elapsed_ticks());
        }
        let expected = micros / 1_000 * 20 / 1_000;
        assert!(
            (ticks - expected).abs() <= 1,
            "period {period} us: {ticks} ticks in {} ms, expected {expected}",
            micros / 1_000
        );
        assert!(
            most <= MAX_CATCH_UP_TICKS,
            "period {period} us: {most} ticks in one frame"
        );
    }
}

/// A hitch shorter than the 500 ms cap loses nothing: its ticks all run in the next frame, and
/// the long-run count is unchanged. Only time beyond the cap is dropped (as vanilla drops it).
#[test]
fn hitches_under_the_cap_are_caught_up_and_longer_ones_drop_only_the_excess() {
    for (hitch, dropped) in [(200_i64, 0_i64), (450, 0), (700, 4), (950, 9)] {
        let mut timer = Timer::new(TICKS_PER_SECOND);
        let mut clock = 0_i64;
        let mut ticks = 0_i64;
        for step in 0..2_000 {
            clock += if step == 1_000 { hitch } else { 5 };
            timer.update_with(clock, clock);
            ticks += i64::from(timer.elapsed_ticks());
        }
        let expected = clock * 20 / 1_000;
        assert_eq!(
            expected - ticks,
            dropped,
            "a {hitch} ms hitch: {ticks} ticks in {clock} ms"
        );
    }
}
