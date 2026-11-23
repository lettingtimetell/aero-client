//! S22 task 4: NBT / protocol round-trip fuzzing.
//!
//! Two drivers over the oracles in [`mc_parity::fuzz`]:
//!
//! * `proptest`, which shrinks a failure to a minimal input — best when something breaks;
//! * the crate's own deterministic driver, seeded by an integer, which reproduces exactly from
//!   the seed printed in the log and needs no regression file to be committed or found.
//!
//! Set `PROPTEST_CASES` to widen the `proptest` runs and `MC_PARITY_FUZZ_ITERS` to lengthen the
//! deterministic one. The CI job in `.github/workflows/ci.yml` does both, which is what makes
//! this task's "in CI" clause true rather than aspirational.
//!
//! S02 and S06 already have `proptest` suites of their own and those are not duplicated here.
//! What is new is (a) a *shared* set of oracles, so the property is written once and reused,
//! (b) coverage of the two parsers this crate introduces — the capture container and the trace
//! container — and (c) the CI wiring.

use mc_nbt::{Nbt, NbtCompound};
use mc_parity::capture::Direction;
use mc_parity::fuzz::{self, Rng, Verdict};
use mc_protocol::State;
use proptest::prelude::*;

fn iterations() -> u32 {
    std::env::var("MC_PARITY_FUZZ_ITERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2_000)
}

#[test]
fn the_deterministic_driver_finds_nothing_and_reproduces_exactly() {
    let iters = iterations();
    let a = fuzz::run(0x5EED_5EED, iters);
    println!("{}", a.summary());
    assert!(a.is_clean(), "property violations:\n{:#?}", a.broken);

    let b = fuzz::run(0x5EED_5EED, iters);
    assert_eq!(a, b, "the seed must fully determine the run");
}

#[test]
fn a_spread_of_seeds_finds_nothing() {
    // Several seeds rather than one long run: independent starting points reach different
    // corners of the input space than one chain does.
    for seed in [0u64, 1, 42, 0xDEAD_BEEF, u64::MAX, 0x1234_5678_9ABC_DEF0] {
        let s = fuzz::run(seed, 400);
        assert!(s.is_clean(), "seed {seed}:\n{:#?}", s.broken);
        assert!(
            s.rejected > 0 && s.held > 0,
            "seed {seed} produced a degenerate run: {}",
            s.summary()
        );
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(
        std::env::var("PROPTEST_CASES").ok().and_then(|v| v.parse().ok()).unwrap_or(256)
    ))]

    /// The NBT reader is total: `Ok` or `Err`, never a panic, on any bytes at all.
    #[test]
    fn nbt_reading_arbitrary_bytes_never_panics(bytes in proptest::collection::vec(any::<u8>(), 0..512)) {
        prop_assert!(!fuzz::nbt_from_bytes(&bytes).is_broken());
    }

    /// A generated NBT value round-trips: encode, decode equal, re-encode byte-identically.
    #[test]
    fn generated_nbt_values_round_trip(seed in any::<u64>(), depth in 0u32..4) {
        let mut rng = Rng::new(seed);
        let value = fuzz::arbitrary_nbt(&mut rng, depth);
        let verdict = fuzz::nbt_value("root", &value);
        prop_assert!(!verdict.is_broken(), "{:?} on {:?}", verdict, value);
    }

    /// Strings containing the two characters Modified UTF-8 encodes differently (hazard H2)
    /// round-trip. `proptest`'s default string strategy rarely produces either.
    #[test]
    fn nbt_strings_with_nul_and_supplementary_characters_round_trip(
        parts in proptest::collection::vec(
            prop_oneof![
                Just("\u{0}".to_owned()),
                Just("\u{10348}".to_owned()),
                Just("\u{7ff}".to_owned()),
                "[a-z]{0,8}",
            ],
            0..8,
        )
    ) {
        let joined: String = parts.concat();
        let verdict = fuzz::nbt_value("root", &Nbt::String(joined.clone()));
        prop_assert!(!verdict.is_broken(), "{:?} on {:?}", verdict, joined);
    }

    /// Every packet decoder, across every state, direction, and an id range that overshoots
    /// both legal ranges, is total.
    #[test]
    fn packet_decoding_arbitrary_bytes_never_panics(
        state_index in 0usize..4,
        clientbound in any::<bool>(),
        id in -8i32..0x60,
        bytes in proptest::collection::vec(any::<u8>(), 0..384),
    ) {
        let state = [State::Handshaking, State::Status, State::Login, State::Play][state_index];
        let direction = if clientbound { Direction::Clientbound } else { Direction::Serverbound };
        prop_assert!(!fuzz::packet_decode_is_total(state, direction, id, &bytes).is_broken());
    }

    /// The capture parser is total, and anything it accepts reaches a fixed point.
    #[test]
    fn capture_parsing_arbitrary_bytes_never_panics(bytes in proptest::collection::vec(any::<u8>(), 0..512)) {
        prop_assert!(!fuzz::capture_from_bytes(&bytes).is_broken());
    }

    /// A *valid* capture with one bit flipped is far likelier to get past the magic and reach
    /// the interesting code than a random blob is.
    #[test]
    fn a_bit_flipped_capture_never_panics(byte in 0usize..4096, bit in 0u8..8) {
        let mut bytes = mc_parity::synthetic::session().encode().expect("encode");
        if byte < bytes.len() {
            bytes[byte] ^= 1 << bit;
        }
        prop_assert!(!fuzz::capture_from_bytes(&bytes).is_broken());
    }

    /// The trace parser, likewise.
    #[test]
    fn trace_parsing_arbitrary_bytes_never_panics(bytes in proptest::collection::vec(any::<u8>(), 0..512)) {
        prop_assert!(!fuzz::trace_from_bytes(&bytes).is_broken());
    }

    /// Every packet the synthetic session contains round-trips from the value side, which is
    /// where losslessness is a real property (a non-canonical VarInt makes the byte-side claim
    /// false, so it is deliberately not asserted).
    #[test]
    fn every_synthetic_packet_round_trips(index in 0usize..1024) {
        let capture = mc_parity::synthetic::session();
        let frames = capture.frames();
        let frame = &frames[index % frames.len()];
        let outcome = mc_parity::replay::decode_frame(frame).expect("the synthetic session decodes");
        let verdict = fuzz::packet_round_trips(&outcome.packet, frame.state, frame.direction);
        prop_assert_eq!(verdict, Verdict::Held);
    }
}

#[test]
fn deeply_nested_nbt_is_refused_rather_than_overflowing_the_stack() {
    // 4,096 levels, eight times vanilla's 512 cap. A reader without a depth limit recurses
    // itself to death here; the point of the test is that this returns rather than crashing.
    let mut value = Nbt::Byte(1);
    for _ in 0..4_096 {
        let mut c = NbtCompound::new();
        c.insert("n", value);
        value = Nbt::Compound(c);
    }
    let encoded = mc_nbt::to_vec("root", &value);
    match encoded {
        Ok(bytes) => {
            let verdict = fuzz::nbt_from_bytes(&bytes);
            assert!(
                matches!(verdict, Verdict::Rejected(_)),
                "4,096 levels must be refused by the depth limit, got {verdict:?}"
            );
        }
        Err(_) => {
            // A writer that refuses to emit it in the first place is an equally good answer.
        }
    }
}
