//! The bug report, "beacon beam keep twitching backwards as it
//! rotates", replayed frame by frame against the real beam math.
//!
//! The `S03PacketTimeUpdate` cadence below is the real one in
//! `crates/mc-parity/captures/spigot-1.8.8-perfworld-vd16.mccap`, a real 1.8.8 server: 35
//! packets, 20 ticks apart, arriving every 1004.5 ms on average, so the server runs at 19.91 TPS
//! while the client's timer runs at exactly 20. The client below runs the real
//! [`Timer`] (vanilla's millisecond-resolution `Timer.updateTimer`) at 360 frames per second,
//! applies every packet that has arrived before the frame's ticks (`runGameLoop` drains its
//! `scheduledTasks` before `runTick`; mc-rs drains in `before_ticks`), ticks the world
//! (`WorldClient.tick`), and samples the beam as `TileEntityBeaconRenderer` does.
//!
//! **Vanilla's own clock** (`getTotalWorldTime() + partialTicks`) turns the beam backwards each
//! time an `S03` pulls the client back a tick. The render clock mc-rs now animates the beam with
//! ([`Environment::render_world_time`]) never does, and equals vanilla exactly whenever the two
//! clocks agree.

use mc_client::sync::Environment;
use mc_render::entity::tile::beacon_beam;
use mc_render::frame::Timer;

/// `(arrival, totalWorldTime)` for every `S03` in the capture: arrival in milliseconds since the
/// capture started (frame timestamps are microseconds; one decimal kept). The world time equals
/// the total time in this capture.
const CAPTURED_S03: [(f64, i64); 35] = [
    (58.5, 28_610),
    (553.0, 28_619),
    (1_561.9, 28_639),
    (2_568.9, 28_659),
    (3_574.1, 28_679),
    (4_583.0, 28_699),
    (5_591.5, 28_719),
    (6_600.0, 28_739),
    (7_600.6, 28_759),
    (8_605.9, 28_779),
    (9_609.8, 28_799),
    (10_612.6, 28_819),
    (11_618.9, 28_839),
    (12_626.8, 28_859),
    (13_632.7, 28_879),
    (14_635.8, 28_899),
    (15_642.2, 28_919),
    (16_646.0, 28_939),
    (17_646.3, 28_959),
    (18_648.5, 28_979),
    (19_652.5, 28_999),
    (20_656.1, 29_019),
    (21_658.7, 29_039),
    (22_662.2, 29_059),
    (23_665.9, 29_079),
    (24_673.6, 29_099),
    (25_681.1, 29_119),
    (26_687.1, 29_139),
    (27_690.2, 29_159),
    (28_695.8, 29_179),
    (29_697.3, 29_199),
    (30_701.6, 29_219),
    (31_703.2, 29_239),
    (32_706.1, 29_259),
    (33_709.5, 29_279),
];

/// The user's 360 Hz display, uncapped-ish.
const FPS: i64 = 360;

/// The inner shell's first corner, as an angle around the beam axis.
/// java: `0.5D + Math.cos(d2 + 2.356194490192345D) * 0.2D` and its `sin` twin.
fn beam_angle(ticks: i64, partial: f32) -> f64 {
    let [x, z] = beacon_beam(ticks, partial, 4, 1.0).inner_corners[0];
    (z - 0.5).atan2(x - 0.5)
}

/// The inner shell's texture offset at the segment's base (`-1.0D + d1`).
fn beam_scroll(ticks: i64, partial: f32) -> f64 {
    beacon_beam(ticks, partial, 4, 1.0).inner_v[0]
}

/// A difference wrapped into `(-period/2, period/2]`.
fn wrapped(delta: f64, period: f64) -> f64 {
    delta - period * (delta / period).round()
}

#[derive(Debug, Default)]
struct Replay {
    frames: usize,
    /// Frames on which the beam turned the wrong way (its angle grows, the beam turns with
    /// `d2 = d0 * 0.025 * -1.5`).
    turned_back: usize,
    /// Frames on which the texture scrolled the wrong way.
    scrolled_back: usize,
    /// The largest wrong-way turn, in degrees.
    worst_back_degrees: f64,
    /// Frames on which the sampled pair was exactly vanilla's.
    exactly_vanilla: usize,
}

/// Replay the capture with the client's ticks offset by `phase_ms` from the packets.
fn replay(phase_ms: f64, total_offset: i64, smoothed: bool) -> Replay {
    let mut timer = Timer::new(20.0);
    timer.reset_clocks(0, 0);
    let mut env = Environment::new();
    let mut next = 0;
    let mut previous: Option<(f64, f64)> = None;
    let mut out = Replay::default();
    for frame in 0..FPS * 34 {
        let micros = frame * 1_000_000 / FPS;
        // java: `System.nanoTime() / 1000000L` — whole milliseconds.
        let millis = micros / 1_000;
        timer.update_with(millis, millis);
        let mut applied = false;
        while next < CAPTURED_S03.len() && CAPTURED_S03[next].0 + phase_ms <= micros as f64 / 1e3 {
            let total = CAPTURED_S03[next].1 + total_offset;
            env.apply_time(total, total);
            applied |= next == 0;
            next += 1;
        }
        for _ in 0..timer.elapsed_ticks() {
            env.tick();
        }
        let partial = timer.render_partial_ticks();
        let vanilla = (env.total_world_time, partial);
        let (ticks, partial) = if smoothed {
            env.render_world_time(partial)
        } else {
            vanilla
        };
        if (ticks, partial) == vanilla {
            out.exactly_vanilla += 1;
        }
        let now = (beam_angle(ticks, partial), beam_scroll(ticks, partial));
        // The first packet moves the clock from 0 to the server's time, a jump vanilla makes
        // too; only count steps between frames that both have the server's time.
        if let (Some((angle, scroll)), false) = (previous, applied || next == 0) {
            out.frames += 1;
            let turn = wrapped(now.0 - angle, std::f64::consts::TAU);
            if turn > 1e-12 {
                out.turned_back += 1;
                out.worst_back_degrees = out.worst_back_degrees.max(turn.to_degrees());
            }
            if wrapped(now.1 - scroll, 1.0) > 1e-12 {
                out.scrolled_back += 1;
            }
        }
        previous = Some(now);
    }
    out
}

#[test]
fn vanillas_clock_turns_the_beam_back_on_the_real_servers_resyncs() {
    let mut turned_back = 0;
    let mut worst = 0.0_f64;
    for phase in (0..50).step_by(5) {
        let run = replay(f64::from(phase), 0, false);
        eprintln!("vanilla clock, phase {phase:2} ms: {run:?}");
        assert_eq!(
            run.turned_back, run.scrolled_back,
            "both come from the same clock"
        );
        turned_back += run.turned_back;
        worst = worst.max(run.worst_back_degrees);
    }
    // 19.91 TPS against 20: about three one-tick corrections in 34 s at every phase.
    assert!(turned_back >= 10, "only {turned_back} backward frames");
    // One tick of `d2 = d0 * 0.025 * -1.5` is 2.15 degrees, less the frame's own advance.
    assert!(
        (1.9..2.2).contains(&worst),
        "worst backward turn {worst} degrees"
    );
}

#[test]
fn the_render_clock_never_turns_the_beam_back_and_is_vanilla_between_resyncs() {
    for phase in (0..50).step_by(5) {
        for total_offset in [0, 3_000_000_000] {
            let run = replay(f64::from(phase), total_offset, true);
            eprintln!("render clock, phase {phase:2} ms, +{total_offset}: {run:?}");
            assert!(run.frames > 12_000);
            assert_eq!(run.turned_back, 0, "phase {phase}");
            assert_eq!(run.scrolled_back, 0, "phase {phase}");
            // Each one-tick correction is absorbed in ten ticks; everywhere else it is
            // vanilla's exact `(getTotalWorldTime(), partialTicks)`.
            assert!(
                run.exactly_vanilla * 100 >= run.frames * 85,
                "only {} of {} frames are vanilla's pair",
                run.exactly_vanilla,
                run.frames
            );
        }
    }
}

/// The brief's first hypothesis, kept as a guard: past 2^24 ticks an `f32` world clock cannot
/// even represent the partial tick, so a beam animated from one would stall and jump. The
/// renderer's `f64` sum (`(double) getTotalWorldTime() + partialTicks`) is strictly monotonic.
#[test]
fn an_f32_world_clock_would_stall_where_the_f64_beam_math_does_not() {
    let total = 50_000_000_i64;
    let (mut f32_flat, mut f64_back) = (0, 0);
    let (mut last_f32, mut last_angle) = (f32::MIN, None::<f64>);
    for step in 0..36 {
        let partial = step as f32 / 36.0;
        let narrowed = total as f32 + partial;
        if narrowed <= last_f32 {
            f32_flat += 1;
        }
        last_f32 = narrowed;
        let angle = beam_angle(total, partial);
        if let Some(last) = last_angle {
            if wrapped(angle - last, std::f64::consts::TAU) >= 0.0 {
                f64_back += 1;
            }
        }
        last_angle = Some(angle);
    }
    assert!(
        f32_flat >= 30,
        "an f32 clock is flat across the whole tick ({f32_flat})"
    );
    assert_eq!(f64_back, 0, "the beacon's f64 math turns every frame");
}
