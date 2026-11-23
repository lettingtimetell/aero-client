//! S22 task 2: physics replay compared against Java-dumped position traces.
//!
//! # What this adds to S13, and what it does not
//!
//! It does not add evidence. S13 already dumped these traces from the real recompiled 1.8.9
//! bytecode under a JDK and already replays them at tolerance `0.0` in
//! `crates/mc-physics/tests/parity.rs`. Those fixtures are consumed here in place; there is no
//! second copy of them in this crate.
//!
//! What it adds is that the comparison now runs through [`mc_parity::trace`]'s **generic**
//! comparator, so the same machinery can be pointed at any other subsystem that acquires a
//! Java reference generator. Three concrete improvements fall out:
//!
//! * the flags byte becomes seven named boolean channels, so a failure says `in_water` rather
//!   than `0b0100011 != 0b0100111`;
//! * every channel reports its worst error, so "bit-exact" is asserted positively
//!   ([`TraceDiff::is_bit_exact`]) rather than inferred from assertions not firing;
//! * the comparator itself is tested for its ability to *fail*, which a parity test that only
//!   ever passes never demonstrates.

use mc_parity::physics_trace::{self, SCENARIOS};
use mc_parity::trace::{Tolerance, Tolerances, Trace, Value};

#[test]
fn every_java_scenario_replays_bit_exactly_through_the_generic_comparator() {
    let mut total_steps = 0usize;
    for name in SCENARIOS {
        let fixture = physics_trace::load(name).expect("S13's committed fixture");
        let actual = physics_trace::replay(&fixture).expect("replay");
        let diff = fixture
            .reference
            .compare(&actual, &Tolerances::exact())
            .expect("compare");

        println!("{}", diff.summary());
        assert!(
            diff.is_clean(),
            "{name} diverged from real Java:\n{:#?}",
            diff.first_divergence
        );
        assert!(
            diff.is_bit_exact(),
            "{name} passed but was not bit-exact — worst errors {:?}",
            diff.worst
        );
        assert_eq!(
            diff.compared_channels, 15,
            "{name}: channels went unchecked"
        );
        total_steps += diff.steps;
    }

    // Pinned so that a regenerated or truncated fixture set cannot quietly shrink the evidence.
    //
    // Note for the record: S13's handoff entry in section 14 says "12 scenarios (4,590 ticks)".
    // The twelve it shipped actually totalled **4,570** - 375 + 280 + 300 + 250 + 260 + 360 +
    // 300 + 260 + 340 + 280 + 565 + 1000. The scenario count and every per-scenario length were
    // as described; only the sum in the prose was 20 out, which is exactly the kind of unchecked
    // number this crate exists to stop drifting.
    //
    // Wave 1 (GAP A11) added ten more scenarios - creative flight, a server-sent movement speed,
    // knockback, Jump Boost, item use, Depth Strider, spectator `noClip`, a dead player, an
    // unloaded column and the cactus/portal contact callbacks - taking the total to **6,783**
    // ticks across 22 scenarios, every one of them still bit-exact at tolerance 0.0.
    assert_eq!(
        total_steps, 6783,
        "the committed fixtures total 6,783 ticks across 22 scenarios"
    );
}

#[test]
fn the_thousand_tick_scenario_is_present_and_is_a_thousand_ticks() {
    let fixture = physics_trace::load("long_random").expect("fixture");
    assert_eq!(
        fixture.reference.len(),
        1000,
        "the plan's S13 acceptance run is 1,000 ticks"
    );
    assert_eq!(fixture.inputs.len(), 1000);
}

#[test]
fn the_reference_traces_survive_the_generic_container() {
    // If a Java generator for another subsystem emits `MCTRACE1` directly, this is the path its
    // output takes. Proving it on 6,783 real steps is worth more than proving it on a fixture
    // written for the purpose.
    for name in SCENARIOS {
        let fixture = physics_trace::load(name).expect("fixture");
        let bytes = fixture.reference.encode();
        let back = Trace::decode(&bytes).expect("decode");
        assert_eq!(
            back.encode(),
            bytes,
            "{name}: encode/decode/encode is not a fixed point"
        );
        assert_eq!(back.name, fixture.reference.name);
        assert_eq!(back.len(), fixture.reference.len());
        let diff = fixture
            .reference
            .compare(&back, &Tolerances::exact())
            .expect("compare");
        assert!(diff.is_bit_exact(), "{name}: the container lost precision");
    }
}

#[test]
fn the_comparator_names_the_exact_step_channel_and_flag_that_diverged() {
    // A parity test that has never been shown to fail is a parity test nobody has checked.
    // Perturb one sample of one channel at one step and assert the report points at it.
    let fixture = physics_trace::load("flat_walk").expect("fixture");
    let good = physics_trace::replay(&fixture).expect("replay");

    // Rebuild `good` with step 100's `in_water` flipped and step 200's `pos.y` moved by 1 ulp.
    let channels = physics_trace::channels();
    let in_water = channels
        .iter()
        .position(|c| c.name == "in_water")
        .expect("in_water channel");
    let pos_y = channels
        .iter()
        .position(|c| c.name == "pos.y")
        .expect("pos.y channel");

    let mut broken = Trace::new(good.name.clone(), channels.clone());
    for step in 0..good.len() {
        let mut row = good.step(step).expect("step").to_vec();
        if step == 100 {
            if let Value::Bool(b) = row[in_water] {
                row[in_water] = Value::Bool(!b);
            }
        }
        if step == 200 {
            if let Value::F64(v) = row[pos_y] {
                row[pos_y] = Value::F64(f64::from_bits(v.to_bits() + 1));
            }
        }
        broken.push(row).expect("row");
    }

    let diff = fixture
        .reference
        .compare(&broken, &Tolerances::exact())
        .expect("compare");
    assert_eq!(
        diff.mismatches, 2,
        "exactly the two perturbations: {diff:?}"
    );
    let first = diff.first_divergence.as_ref().expect("a divergence");
    assert_eq!(first.step, 100);
    assert_eq!(first.channel, "in_water");
    assert_eq!(diff.mismatches_by_channel.get("pos.y"), Some(&1));
    assert_eq!(diff.mismatches_by_channel.get("in_water"), Some(&1));

    // And the same perturbation must still be caught at 1e-9, the tolerance the plan asks for —
    // one ulp of a value around y=64 is about 1.4e-14, so a loose-enough tolerance *would*
    // swallow it. That is exactly why the default is `Exact`.
    let at_1e9 = fixture
        .reference
        .compare(&broken, &Tolerances::uniform(Tolerance::Abs(1e-9)))
        .expect("compare");
    assert_eq!(
        at_1e9.mismatches, 1,
        "at 1e-9 the flag change is still caught but the one-ulp position change is not — \
         which is the argument for bit-exact comparison, made numerically"
    );
}

#[test]
fn a_trace_compared_against_a_shorter_one_is_a_shape_error_not_a_pass() {
    let fixture = physics_trace::load("flat_walk").expect("fixture");
    let mut short = Trace::new("flat_walk", physics_trace::channels());
    short
        .push(fixture.reference.step(0).expect("step 0").to_vec())
        .expect("row");
    assert!(
        fixture
            .reference
            .compare(&short, &Tolerances::exact())
            .is_err(),
        "a one-step trace must not silently 'match' a 300-step reference"
    );
}

#[test]
fn the_scenarios_still_exercise_water_lava_ladders_and_collisions() {
    // The comparator only proves the two sides agree. If a fixture regenerated into "the player
    // stands still for 300 ticks", the agreement would be perfect and meaningless. S13 guards
    // this in its own test; the same guard belongs next to any reuse of the fixtures.
    let has_true = |name: &str, channel: &str| -> usize {
        let fixture = physics_trace::load(name).expect("fixture");
        fixture
            .reference
            .column(channel)
            .expect("channel")
            .iter()
            .filter(|v| **v == Value::Bool(true))
            .count()
    };
    assert!(has_true("water_swim", "in_water") > 100);
    assert!(has_true("lava_swim", "in_lava") > 100);
    assert!(has_true("ladder_climb", "on_ladder") > 100);
    assert!(has_true("wall_collide", "collided_horizontally") > 50);
    assert!(has_true("flat_walk", "sprinting") > 0);
    assert!(has_true("long_random", "in_water") > 100);
}
