//! A shared fuzz harness over NBT, protocol packets, and capture files.
//!
//! # Why this is a library module and not just a `proptest` file
//!
//! `proptest` is a dev-dependency, so anything written inside a `proptest!` block can only ever
//! run under `cargo test`. The *oracles* — the functions that say whether a property held —
//! belong in the library, where three different drivers can share them:
//!
//! 1. `tests/fuzz.rs` drives them with `proptest`, which shrinks a failure to a minimal input;
//! 2. [`run`] drives them from a **deterministic** PRNG seeded by an integer, so a CI failure
//!    reproduces exactly from the seed printed in the log, with no `proptest` regression file
//!    to find;
//! 3. a future `cargo-fuzz` or AFL target can call them on a coverage-guided corpus without
//!    this crate growing a dependency on either.
//!
//! # The three properties
//!
//! * **Totality.** Every parser in reach of untrusted bytes returns `Ok` or `Err`, never a
//!   panic. `mc-protocol` already fuzzes its own decoders this way (S06); what is new here is
//!   that the *capture* and *trace* parsers this crate introduces get the same treatment, and
//!   that all of them run under one driver.
//! * **Round-trip stability.** A value that encodes and decodes must come back equal, and
//!   re-encoding must be byte-identical. Note this is asserted from the **value** side, not
//!   from random bytes: a non-canonical VarInt (`0x80 0x00` is a legal encoding of zero) decodes
//!   fine and re-encodes shorter, so "random bytes re-encode to themselves" is simply false and
//!   asserting it would be a bug in the harness rather than in the code under test.
//! * **Idempotence of the capture container.** Parse → encode → parse must reach a fixed point.

use mc_nbt::{Nbt, NbtCompound, NbtLimits};
use mc_protocol::PacketBuffer;

use crate::capture::{Capture, Direction};
use crate::replay::{decode_body, Decoded};

/// What an oracle concluded about one input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    /// The input was refused cleanly. For malformed input this is the healthy answer, not a
    /// failure — a fuzz run in which nothing is ever rejected is a fuzz run that is not
    /// reaching the parser.
    Rejected(String),
    /// The input was accepted and the property held.
    Held,
    /// The input was accepted but the property did **not** hold. This is the only outcome that
    /// is a bug.
    Broken(String),
}

impl Verdict {
    /// Whether this verdict is a bug.
    #[must_use]
    pub fn is_broken(&self) -> bool {
        matches!(self, Verdict::Broken(_))
    }

    /// Whether the input was accepted (held or broken, but not rejected).
    #[must_use]
    pub fn was_accepted(&self) -> bool {
        !matches!(self, Verdict::Rejected(_))
    }
}

/// Decode NBT from untrusted bytes and, if it decodes, check that it round-trips.
///
/// Structural equality rather than byte equality on the first hop, because vanilla itself is
/// not byte-idempotent here: `NBTTagList.write` recomputes an empty list's element type and
/// writes `TAG_End`, so an empty typed list does not survive a Java round trip either
/// (`mc_nbt::tag::Nbt::List` documents this). The *second* hop must be byte-stable, and that is
/// checked too — once written by us, the encoding is canonical and must stay put.
#[must_use]
pub fn nbt_from_bytes(bytes: &[u8]) -> Verdict {
    let (name, value) = match mc_nbt::from_slice_with_limits(bytes, NbtLimits::PROTOCOL) {
        Ok(v) => v,
        Err(e) => return Verdict::Rejected(e.to_string()),
    };
    nbt_value(&name, &value)
}

/// Check that a value encodes, decodes back equal, and re-encodes byte-identically.
#[must_use]
pub fn nbt_value(name: &str, value: &Nbt) -> Verdict {
    let first = match mc_nbt::to_vec(name, value) {
        Ok(v) => v,
        Err(e) => return Verdict::Rejected(format!("write refused the value: {e}")),
    };
    let (name2, value2) = match mc_nbt::from_slice_with_limits(&first, NbtLimits::UNLIMITED) {
        Ok(v) => v,
        Err(e) => {
            return Verdict::Broken(format!("our own encoding did not decode: {e}"));
        }
    };
    if name2 != name {
        return Verdict::Broken(format!("root name `{name}` came back as `{name2}`"));
    }
    let second = match mc_nbt::to_vec(&name2, &value2) {
        Ok(v) => v,
        Err(e) => return Verdict::Broken(format!("re-encoding a decoded value failed: {e}")),
    };
    if first != second {
        return Verdict::Broken(format!(
            "encoding is not stable: {} bytes then {} bytes",
            first.len(),
            second.len()
        ));
    }
    if !nbt_bit_eq(value, &value2) {
        return Verdict::Broken("the decoded value differs from the encoded one".to_owned());
    }
    Verdict::Held
}

/// Structural equality over NBT with **bit-pattern** comparison for floats.
///
/// `Nbt`'s derived `PartialEq` uses `f32`/`f64` `==`, under which `NaN != NaN` and
/// `-0.0 == 0.0`. Both are wrong for a round-trip oracle, and in opposite directions: the first
/// reports a perfect round trip of a NaN as a failure, and the second would wave through a lost
/// sign bit. The rest of this crate compares by bits for exactly these reasons
/// ([`crate::trace::Tolerance::Exact`]); NBT gets the same treatment.
///
/// This used to duplicate the comparison inline; it now
/// delegates to [`Nbt::bit_eq`], which is where the round-trip guarantee actually belongs.
#[must_use]
pub fn nbt_bit_eq(a: &Nbt, b: &Nbt) -> bool {
    a.bit_eq(b)
}

/// Feed a `(state, direction, id)` triple arbitrary bytes and confirm the decoder is total.
///
/// The property is only "no panic": a decoder that leaves bytes unread has not necessarily
/// misbehaved, since the random tail was never a packet in the first place. Losslessness is
/// asserted from the value side by [`packet_round_trips`] instead.
#[must_use]
pub fn packet_decode_is_total(
    state: mc_protocol::State,
    direction: Direction,
    id: i32,
    bytes: &[u8],
) -> Verdict {
    let mut buf = PacketBuffer::from_slice(bytes);
    match decode_body(state, direction, id, &mut buf) {
        Ok(_) => Verdict::Held,
        Err(e) => Verdict::Rejected(e.to_string()),
    }
}

/// Check that a decoded packet encodes and decodes back to an equal value, byte-identically.
#[must_use]
pub fn packet_round_trips(
    packet: &Decoded,
    state: mc_protocol::State,
    direction: Direction,
) -> Verdict {
    let bytes = packet.encode();
    let Some((id, used)) = crate::capture::read_varint(&bytes) else {
        return Verdict::Broken("an encoded packet had no readable id varint".to_owned());
    };
    let mut buf = PacketBuffer::from_slice(&bytes[used..]);
    let decoded = match decode_body(state, direction, id, &mut buf) {
        Ok(d) => d,
        Err(e) => return Verdict::Broken(format!("our own encoding did not decode: {e}")),
    };
    if buf.remaining() != 0 {
        return Verdict::Broken(format!(
            "{} left {} bytes of its own encoding unread",
            packet.kind(),
            buf.remaining()
        ));
    }
    if decoded != *packet {
        return Verdict::Broken(format!("{} did not survive a round trip", packet.kind()));
    }
    if decoded.encode() != bytes {
        return Verdict::Broken(format!("{}'s encoding is not stable", packet.kind()));
    }
    Verdict::Held
}

/// Parse a capture from untrusted bytes and, if it parses, check that it reaches a fixed point.
#[must_use]
pub fn capture_from_bytes(bytes: &[u8]) -> Verdict {
    let capture = match Capture::parse(bytes) {
        Ok(c) => c,
        Err(e) => return Verdict::Rejected(e.to_string()),
    };
    let reencoded = match capture.encode() {
        Ok(v) => v,
        Err(e) => return Verdict::Broken(format!("a parsed capture would not re-encode: {e}")),
    };
    match Capture::parse(&reencoded) {
        Ok(again) if again == capture => Verdict::Held,
        Ok(_) => Verdict::Broken("parse/encode/parse did not reach a fixed point".to_owned()),
        Err(e) => Verdict::Broken(format!("our own capture encoding did not parse: {e}")),
    }
}

/// Parse a trace from untrusted bytes and check the same fixed point.
#[must_use]
pub fn trace_from_bytes(bytes: &[u8]) -> Verdict {
    let trace = match crate::trace::Trace::decode(bytes) {
        Ok(t) => t,
        Err(e) => return Verdict::Rejected(e.to_string()),
    };
    let reencoded = trace.encode();
    match crate::trace::Trace::decode(&reencoded) {
        Ok(again) if again.encode() == reencoded => Verdict::Held,
        Ok(_) => Verdict::Broken("decode/encode/decode did not reach a fixed point".to_owned()),
        Err(e) => Verdict::Broken(format!("our own trace encoding did not decode: {e}")),
    }
}

/// The highest packet id defined for a `(state, direction)` pair.
///
/// Written out rather than derived, so it is a second, independent statement of the id space;
/// `mc-protocol`'s own test asserts the Play ranges are exactly `0x00..=0x49` and `0x00..=0x19`,
/// and if either ever changed, this constant and that test would have to be updated together.
#[must_use]
pub const fn legal_id_ceiling(state: mc_protocol::State, direction: Direction) -> u32 {
    use mc_protocol::State;
    match (state, direction) {
        (State::Handshaking, Direction::Serverbound) => 0x00,
        (State::Handshaking, Direction::Clientbound) => 0x00,
        (State::Status, _) => 0x01,
        (State::Login, Direction::Serverbound) => 0x01,
        (State::Login, Direction::Clientbound) => 0x03,
        (State::Play, Direction::Clientbound) => 0x49,
        (State::Play, Direction::Serverbound) => 0x19,
    }
}

/// A small, fast, deterministic PRNG (SplitMix64).
///
/// Not cryptographic and not Java-compatible — [`mc_math`] has the Java LCG for the places
/// where matching vanilla's random sequence matters. This one exists purely so a fuzz run is
/// reproducible from a printed seed.
#[derive(Debug, Clone)]
pub struct Rng {
    state: u64,
}

impl Rng {
    /// Seed the generator.
    #[must_use]
    pub fn new(seed: u64) -> Rng {
        Rng { state: seed }
    }

    /// The next 64 bits.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// The next 32 bits.
    pub fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }

    /// A value in `0..n`. `n == 0` yields `0`.
    pub fn below(&mut self, n: u32) -> u32 {
        if n == 0 {
            0
        } else {
            self.next_u32() % n
        }
    }

    /// A coin flip.
    pub fn boolean(&mut self) -> bool {
        self.next_u64() & 1 == 1
    }

    /// `len` random bytes.
    pub fn bytes(&mut self, len: usize) -> Vec<u8> {
        (0..len).map(|_| self.next_u32() as u8).collect()
    }
}

/// Generate an arbitrary NBT value, at most `depth` levels of nesting deep.
///
/// Biased towards the shapes that break things: empty and single-element lists, compounds with
/// duplicate-looking keys, `NaN`/`-0.0` floats, and strings containing the characters Java's
/// Modified UTF-8 encodes differently from standard UTF-8 (hazard **H2**) — a `NUL`, which
/// becomes two bytes, and a supplementary character, which becomes a CESU-8 surrogate pair.
pub fn arbitrary_nbt(rng: &mut Rng, depth: u32) -> Nbt {
    const TRICKY_STRINGS: [&str; 6] = [
        "",
        "plain",
        "\u{0}embedded nul",
        "\u{10348}",            // supplementary: CESU-8 surrogate pair in Modified UTF-8
        "\u{7f}\u{80}\u{7ff}",  // the 1/2-byte boundary in both encodings
        "§aformatting §rcodes", // the section sign, everywhere in real chat data
    ];
    let leaf = depth == 0;
    let choice = rng.below(if leaf { 9 } else { 11 });
    match choice {
        0 => Nbt::Byte(rng.next_u32() as i8),
        1 => Nbt::Short(rng.next_u32() as i16),
        2 => Nbt::Int(rng.next_u32() as i32),
        3 => Nbt::Long(rng.next_u64() as i64),
        4 => Nbt::Float(match rng.below(4) {
            0 => f32::NAN,
            1 => -0.0,
            2 => f32::INFINITY,
            _ => f32::from_bits(rng.next_u32()),
        }),
        5 => Nbt::Double(match rng.below(4) {
            0 => f64::NAN,
            1 => -0.0,
            2 => f64::NEG_INFINITY,
            _ => f64::from_bits(rng.next_u64()),
        }),
        6 => {
            let len = rng.below(17) as usize;
            Nbt::ByteArray(rng.bytes(len))
        }
        7 => {
            Nbt::String(TRICKY_STRINGS[rng.below(TRICKY_STRINGS.len() as u32) as usize].to_owned())
        }
        8 => Nbt::IntArray((0..rng.below(9)).map(|_| rng.next_u32() as i32).collect()),
        9 => {
            // A list must be homogeneous, so one element decides the type and the rest follow.
            let len = rng.below(5);
            if len == 0 {
                return Nbt::list(Vec::new());
            }
            let first = arbitrary_nbt(rng, depth - 1);
            let ty = first.tag_type();
            let mut items = vec![first];
            while items.len() < len as usize {
                let candidate = arbitrary_nbt(rng, depth - 1);
                if candidate.tag_type() == ty {
                    items.push(candidate);
                } else {
                    items.push(items[0].clone());
                }
            }
            Nbt::list(items)
        }
        _ => {
            let mut c = NbtCompound::new();
            for i in 0..rng.below(5) {
                c.insert(format!("k{i}"), arbitrary_nbt(rng, depth - 1));
            }
            Nbt::Compound(c)
        }
    }
}

/// What a [`run`] found.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FuzzSummary {
    /// The seed this run used. Print it: it is the whole reproduction recipe.
    pub seed: u64,
    /// How many iterations ran.
    pub iterations: u32,
    /// Inputs accepted by a parser and satisfying its property.
    pub held: u32,
    /// Inputs cleanly refused.
    pub rejected: u32,
    /// Every property violation found, described.
    pub broken: Vec<String>,
    /// How many random-byte inputs the NBT parser accepted. A run where this is zero has not
    /// actually reached the parser and its "no failures" result means nothing.
    pub nbt_accepted_random_bytes: u32,
    /// How many random-byte inputs a packet decoder accepted, likewise.
    pub packets_accepted_random_bytes: u32,
}

impl FuzzSummary {
    /// Whether the run found no property violations.
    #[must_use]
    pub fn is_clean(&self) -> bool {
        self.broken.is_empty()
    }

    /// A one-line summary suitable for a test's stdout.
    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "seed {}: {} iterations, {} held, {} rejected, {} broken; \
             random bytes accepted: {} nbt, {} packets",
            self.seed,
            self.iterations,
            self.held,
            self.rejected,
            self.broken.len(),
            self.nbt_accepted_random_bytes,
            self.packets_accepted_random_bytes
        )
    }
}

/// Run the whole harness deterministically.
///
/// Each iteration does five things: throws random bytes at the NBT reader, round-trips a
/// generated NBT value, throws random bytes at a randomly chosen `(state, direction, id)`
/// packet decoder, round-trips one of the synthetic session's real packets, and throws
/// mutated bytes at the capture and trace parsers.
///
/// A panic anywhere in there fails the caller's test, which is the point: totality is asserted
/// by the absence of a panic, not by a return value.
#[must_use]
pub fn run(seed: u64, iterations: u32) -> FuzzSummary {
    let mut rng = Rng::new(seed);
    let mut summary = FuzzSummary {
        seed,
        iterations,
        ..FuzzSummary::default()
    };

    let session = crate::synthetic::session();
    let known_packets: Vec<(mc_protocol::State, Direction, Decoded)> = session
        .frames()
        .iter()
        .filter_map(|f| {
            crate::replay::decode_frame(f)
                .ok()
                .map(|o| (f.state, f.direction, o.packet))
        })
        .collect();
    let valid_capture = session.encode_with(false).unwrap_or_default();
    let valid_trace = crate::trace::Trace::new("fuzz", Vec::new()).encode();

    let states = [
        mc_protocol::State::Handshaking,
        mc_protocol::State::Status,
        mc_protocol::State::Login,
        mc_protocol::State::Play,
    ];

    let record = |summary: &mut FuzzSummary, verdict: Verdict, what: &str| match verdict {
        Verdict::Held => summary.held += 1,
        Verdict::Rejected(_) => summary.rejected += 1,
        Verdict::Broken(why) => summary.broken.push(format!("{what}: {why}")),
    };

    for _ in 0..iterations {
        // 1. Random bytes at the NBT reader.
        let len = rng.below(96) as usize;
        let bytes = rng.bytes(len);
        let verdict = nbt_from_bytes(&bytes);
        if verdict.was_accepted() {
            summary.nbt_accepted_random_bytes += 1;
        }
        record(&mut summary, verdict, "nbt from random bytes");

        // 2. A generated NBT value must round-trip.
        let value = arbitrary_nbt(&mut rng, 3);
        record(
            &mut summary,
            nbt_value("root", &value),
            "generated nbt value",
        );

        // 3. Random bytes at a random packet decoder.
        let state = states[rng.below(states.len() as u32) as usize];
        let direction = if rng.boolean() {
            Direction::Clientbound
        } else {
            Direction::Serverbound
        };
        // Half the ids are drawn from inside the legal range for this (state, direction), so
        // the run actually reaches the decoders; the other half sweep past both ends, because
        // an id no packet claims must produce a clean `UnknownPacketId` and the dispatch
        // macro's match arms are exactly where an off-by-one hides. Drawing uniformly from
        // `-4..0x5C` alone left ~93% of iterations bouncing off `UnknownPacketId` — a fuzz run
        // that never enters the code it is fuzzing.
        let id = if rng.boolean() {
            rng.below(legal_id_ceiling(state, direction) + 1) as i32
        } else {
            rng.below(0x60) as i32 - 4
        };
        let body_len = rng.below(160) as usize;
        let body = rng.bytes(body_len);
        let verdict = packet_decode_is_total(state, direction, id, &body);
        if verdict.was_accepted() {
            summary.packets_accepted_random_bytes += 1;
        }
        record(&mut summary, verdict, "packet from random bytes");

        // 4. A real packet must round-trip.
        if !known_packets.is_empty() {
            let (state, direction, packet) =
                &known_packets[rng.below(known_packets.len() as u32) as usize];
            record(
                &mut summary,
                packet_round_trips(packet, *state, *direction),
                "known packet round trip",
            );
        }

        // 5. Mutated container bytes. Half the time a fresh random blob, half the time a valid
        //    capture with one byte flipped — the second is far likelier to get past the magic
        //    and reach the interesting code.
        let mutated = if rng.boolean() || valid_capture.is_empty() {
            let len = rng.below(128) as usize;
            rng.bytes(len)
        } else {
            let mut copy = valid_capture.clone();
            let at = rng.below(copy.len() as u32) as usize;
            copy[at] ^= 1 << rng.below(8);
            copy
        };
        record(&mut summary, capture_from_bytes(&mutated), "capture parse");

        let mutated_trace = if rng.boolean() {
            let len = rng.below(96) as usize;
            rng.bytes(len)
        } else {
            let mut copy = valid_trace.clone();
            let at = rng.below(copy.len() as u32) as usize;
            copy[at] ^= 1 << rng.below(8);
            copy
        };
        record(
            &mut summary,
            trace_from_bytes(&mutated_trace),
            "trace parse",
        );
    }

    summary
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_short_deterministic_run_is_clean_and_reproducible() {
        let a = run(0xC0FFEE, 200);
        let b = run(0xC0FFEE, 200);
        println!("{}", a.summary());
        assert!(a.is_clean(), "{:#?}", a.broken);
        assert_eq!(a, b, "the same seed must produce the same run");
    }

    #[test]
    fn a_different_seed_produces_a_different_run() {
        let a = run(1, 100);
        let b = run(2, 100);
        assert_ne!(
            (a.held, a.rejected, a.nbt_accepted_random_bytes),
            (b.held, b.rejected, b.nbt_accepted_random_bytes),
            "two seeds producing identical statistics suggests the seed is being ignored"
        );
    }

    #[test]
    fn the_run_actually_reaches_the_parsers_it_claims_to() {
        // A fuzz run where every input bounces off a magic-number check proves nothing. These
        // floors are what make "no failures" meaningful.
        let s = run(7, 500);
        assert!(
            s.nbt_accepted_random_bytes > 0,
            "no random byte string was ever valid NBT: {}",
            s.summary()
        );
        assert!(
            s.packets_accepted_random_bytes > 100,
            "random bytes almost never decoded as a packet: {}",
            s.summary()
        );
        assert!(
            s.rejected > 100,
            "nothing was being rejected: {}",
            s.summary()
        );
        assert!(s.held > 500, "nothing was holding: {}", s.summary());
    }

    #[test]
    fn the_generator_produces_every_tag_type() {
        let mut rng = Rng::new(42);
        let mut seen = std::collections::BTreeSet::new();
        for _ in 0..2000 {
            seen.insert(arbitrary_nbt(&mut rng, 3).tag_type());
        }
        // 11 of the 12 types; `TAG_End` is never a value, only an empty list's element type.
        assert_eq!(
            seen.len(),
            11,
            "the generator only ever emitted {seen:?}, which would leave tag types unfuzzed"
        );
    }

    #[test]
    fn the_nbt_oracle_catches_a_value_that_cannot_round_trip() {
        // `Nbt::End` inside a compound is exactly the value `write` is documented to refuse.
        let mut c = NbtCompound::new();
        c.insert("bad", Nbt::End);
        let verdict = nbt_value("root", &Nbt::Compound(c));
        assert!(
            matches!(verdict, Verdict::Rejected(_)),
            "expected a clean refusal, got {verdict:?}"
        );
    }

    #[test]
    fn the_nbt_oracle_compares_floats_by_bits_not_by_partial_eq() {
        // Both of these are why the oracle cannot use `Nbt`'s derived `PartialEq`. The first
        // was a real false positive found by the fuzz driver on its first run.
        assert!(
            Nbt::Double(f64::NAN) != Nbt::Double(f64::NAN),
            "the derived PartialEq says a NaN differs from itself"
        );
        assert!(
            nbt_bit_eq(&Nbt::Double(f64::NAN), &Nbt::Double(f64::NAN)),
            "but an identical bit pattern is a perfect round trip"
        );

        assert!(
            Nbt::Float(0.0) == Nbt::Float(-0.0),
            "the derived PartialEq says +0 and -0 are the same value"
        );
        assert!(
            !nbt_bit_eq(&Nbt::Float(0.0), &Nbt::Float(-0.0)),
            "but a lost sign bit must never pass as a round trip"
        );

        // And it recurses, so a NaN buried in a list inside a compound still works.
        let mut a = NbtCompound::new();
        a.insert("l", Nbt::list(vec![Nbt::Double(f64::NAN)]));
        let mut b = NbtCompound::new();
        b.insert("l", Nbt::list(vec![Nbt::Double(f64::NAN)]));
        assert!(nbt_bit_eq(&Nbt::Compound(a), &Nbt::Compound(b)));
    }

    #[test]
    fn every_legal_id_ceiling_is_actually_the_last_defined_id() {
        use mc_protocol::State;
        for (state, direction) in [
            (State::Handshaking, Direction::Serverbound),
            (State::Status, Direction::Serverbound),
            (State::Status, Direction::Clientbound),
            (State::Login, Direction::Serverbound),
            (State::Login, Direction::Clientbound),
            (State::Play, Direction::Clientbound),
            (State::Play, Direction::Serverbound),
        ] {
            let ceiling = legal_id_ceiling(state, direction) as i32;
            // The ceiling must decode (given enough bytes it may still fail on content, but it
            // must not be `UnknownPacketId`), and one past it must not exist at all.
            let plenty = vec![0u8; 512];
            let mut buf = PacketBuffer::from_slice(&plenty);
            let at = decode_body(state, direction, ceiling, &mut buf);
            assert!(
                !matches!(at, Err(mc_protocol::ProtoError::UnknownPacketId { .. })),
                "{state:?}/{direction:?} id {ceiling:#04x} is not a defined packet"
            );
            let mut buf = PacketBuffer::from_slice(&plenty);
            let past = decode_body(state, direction, ceiling + 1, &mut buf);
            assert!(
                matches!(past, Err(mc_protocol::ProtoError::UnknownPacketId { .. })),
                "{state:?}/{direction:?} id {:#04x} exists but the ceiling says it should not",
                ceiling + 1
            );
        }
    }

    #[test]
    fn a_valid_capture_reaches_a_fixed_point() {
        let bytes = crate::synthetic::session().encode().expect("encode");
        assert_eq!(capture_from_bytes(&bytes), Verdict::Held);
    }
}
