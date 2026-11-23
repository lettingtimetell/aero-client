//! S22 task 1: packet-capture recorder + deterministic replay.
//!
//! Three things are asserted here, in increasing strength:
//!
//! 1. the container survives a round trip and rejects malformed input (unit-tested in
//!    `src/capture.rs`; here it is exercised on a whole session and on a real file on disk);
//! 2. the **stream** recorder reconstructs the identical frame sequence from a byte stream cut
//!    at every awkward boundary, which is the part that has to work when a live socket is on
//!    the other end;
//! 3. replay is deterministic and lossless — every frame decodes, and re-encoding the decoded
//!    value reproduces the captured bytes exactly.
//!
//! # The gap S22 recorded, and how much of it is now closed
//!
//! Most captures replayed here were **built by this workspace's own encoders**, so for those,
//! point 3 says "the decoder is the encoder's exact inverse", not "a real server's bytes
//! decode". `a_real_capture_replays_clean_when_one_exists` was left as the seam for the day a
//! genuine recording existed.
//!
//! **That day arrived** (S21c, 2026-08-15). `tests/fixtures/` holds two real sessions against a
//! real Spigot 1.8.8 server on protocol 47, both recorded by `mc-client`'s own connection task
//! through `mc_parity::StreamRecorder`:
//!
//! | file | frames | distinct ids | how it was produced |
//! |---|---|---|---|
//! | `spigot-1.8.8-offline-join-walk-chat.mccap` | 898 | 37 | a scripted 30 s walk/sprint/sneak/jump/chat run of `mc-client` against a local server |
//! | `spigot-1.8.8-offline-binary-session.mccap` | 1679 | 38 | the shipping `mc-client.exe`, launched with launcher credentials and `MC_CAPTURE` |
//!
//! The test below replays **every** `.mccap` in that directory by default, so committing a new
//! recording is all it takes to have it checked. `MC_PARITY_CAPTURE` still overrides the set with
//! a single file.
//!
//! Two limits on what that fixture proves, kept next to it so they cannot drift from the claim:
//! the server was **Spigot, not vanilla** (Spigot rewrites chunk data for anti-xray), and it was
//! in **offline mode**, so the capture contains no encrypted traffic and the session-server join
//! never happened. Compression *was* negotiated at threshold 256, so the recording does span both
//! sides of the zlib threshold.

use mc_parity::capture::{frame_bytes, Capture, CaptureLimits, CaptureMeta, StreamRecorder};
use mc_parity::{synthetic, Direction, Replayer};

#[test]
fn the_synthetic_session_replays_losslessly_and_deterministically() {
    let capture = synthetic::session();
    let first = Replayer::new(&capture).run();
    println!("{}", first.summary());

    assert!(
        first.is_clean(),
        "every frame must decode and re-encode byte-identically:\n{first:#?}"
    );
    assert_eq!(first.frames, capture.len());
    assert_eq!(first.byte_identical, capture.len());
    assert!(
        first.distinct_packets() >= 10,
        "a session exercising fewer than ten distinct packets is not a session: {}",
        first.summary()
    );

    let second = Replayer::new(&capture).run();
    assert_eq!(
        first, second,
        "replay must not depend on a clock, a hash order, or anything else"
    );
}

#[test]
fn replay_can_be_restricted_to_one_direction() {
    let capture = synthetic::session();
    let clientbound = Replayer::new(&capture).only(Direction::Clientbound).run();
    let serverbound = Replayer::new(&capture).only(Direction::Serverbound).run();
    assert!(clientbound.is_clean() && serverbound.is_clean());
    assert_eq!(
        clientbound.frames + serverbound.frames,
        capture.len(),
        "the two halves must partition the capture"
    );
    assert!(clientbound.frames > 0 && serverbound.frames > 0);
}

#[test]
fn the_capture_container_round_trips_a_whole_session_compressed_and_not() {
    let capture = synthetic::session();
    let compressed = capture.encode_with(true).expect("compressed encode");
    let plain = capture.encode_with(false).expect("plain encode");

    assert_eq!(
        Capture::parse(&compressed).expect("parse compressed"),
        capture
    );
    assert_eq!(Capture::parse(&plain).expect("parse plain"), capture);
    assert!(
        compressed.len() < plain.len(),
        "zlib should shrink a session full of repeated movement packets: {} vs {}",
        compressed.len(),
        plain.len()
    );
}

#[test]
fn a_capture_survives_a_trip_through_the_filesystem() {
    let capture = synthetic::session();
    let path = std::env::temp_dir().join(format!(
        "mc-parity-{}-{}.mccap",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos())
    ));
    capture.write_file(&path).expect("write");
    let back = Capture::read_file(&path).expect("read");
    let _ = std::fs::remove_file(&path);
    assert_eq!(back, capture);
}

#[test]
fn the_stream_recorder_reconstructs_the_session_from_a_byte_stream() {
    let expected = synthetic::session();
    let stream = synthetic::session_stream();

    // Every chunk size that could plausibly split a length prefix, a small packet, or a large
    // one — plus 1, which splits everything.
    for chunk in [1usize, 2, 3, 5, 8, 13, 64, 255, 1024, usize::MAX] {
        let mut rec = StreamRecorder::new(CaptureMeta {
            protocol: 47,
            address: synthetic::SYNTHETIC_ADDRESS.to_owned(),
            started_at_unix_millis: 1_700_000_000_000,
        });
        let mut recorded = 0usize;
        for (i, (direction, state, bytes)) in stream.iter().enumerate() {
            rec.set_state(*state);
            let size = chunk.min(bytes.len()).max(1);
            for part in bytes.chunks(size) {
                recorded += rec
                    .push_at(*direction, expected.frames()[i].micros, part)
                    .expect("push");
            }
        }
        assert_eq!(recorded, stream.len(), "chunk size {chunk}");
        assert_eq!(
            rec.pending(Direction::Clientbound) + rec.pending(Direction::Serverbound),
            0,
            "chunk size {chunk}: bytes left buffered"
        );

        let got = rec.finish();
        assert_eq!(
            got, expected,
            "chunk size {chunk}: the stream recorder must rebuild the identical capture"
        );
    }
}

#[test]
fn a_stream_recording_replays_exactly_as_the_original_does() {
    let stream = synthetic::session_stream();
    let mut rec = StreamRecorder::new(CaptureMeta::new(synthetic::SYNTHETIC_ADDRESS));
    for (direction, state, bytes) in &stream {
        rec.set_state(*state);
        rec.push_at(*direction, 0, bytes).expect("push");
    }
    let report = Replayer::new(&rec.finish()).run();
    println!("from stream: {}", report.summary());
    assert!(report.is_clean(), "{report:#?}");
}

#[test]
fn a_truncated_stream_leaves_the_partial_frame_out_rather_than_inventing_one() {
    let mut rec = StreamRecorder::new(CaptureMeta::new("x"));
    let complete = frame_bytes(&[0x00, 0x01, 0x02]);
    let partial = frame_bytes(&[0x00, 0xAA, 0xBB, 0xCC]);
    rec.push_at(Direction::Clientbound, 0, &complete)
        .expect("a");
    rec.push_at(Direction::Clientbound, 1, &partial[..3])
        .expect("b");
    // `partial` is a 1-byte length prefix plus 4 payload bytes; three of those five arrived.
    assert_eq!(rec.pending(Direction::Clientbound), 3);
    let capture = rec.finish();
    assert_eq!(
        capture.len(),
        1,
        "the half-arrived frame was never a packet"
    );
}

#[test]
fn the_parser_refuses_a_hostile_file_instead_of_allocating_for_it() {
    let capture = synthetic::session();
    let bytes = capture.encode().expect("encode");

    // A limit set below what this capture needs must be enforced, not ignored.
    let tight = CaptureLimits {
        max_frames: 2,
        ..CaptureLimits::default()
    };
    assert!(
        Capture::parse_with(&bytes, tight).is_err(),
        "a frame-count ceiling below the declared count must reject"
    );

    // Every single-bit flip of a valid capture must either parse or error — never panic, and
    // never produce something that then fails to re-encode.
    for byte in 0..bytes.len().min(400) {
        for bit in 0..8 {
            let mut mutated = bytes.clone();
            mutated[byte] ^= 1 << bit;
            if let Ok(parsed) = Capture::parse(&mutated) {
                let reencoded = parsed.encode().expect("a parsed capture must re-encode");
                let again = Capture::parse(&reencoded).expect("and re-parse");
                assert_eq!(again, parsed, "byte {byte} bit {bit}: not a fixed point");
            }
        }
    }
}

/// Every real recording to replay: `MC_PARITY_CAPTURE` if it is set, otherwise every `.mccap`
/// committed under `tests/fixtures/`.
///
/// Enumerating the directory rather than naming one file means whoever records a
/// session — against Paper, against BungeeCord, against an online-mode server — gets it replayed
/// by simply committing it, with no code change. Sorted so a failure names the same file on
/// every machine.
fn real_captures() -> Vec<std::path::PathBuf> {
    if let Some(path) = std::env::var_os("MC_PARITY_CAPTURE") {
        return vec![std::path::PathBuf::from(path)];
    }
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures");
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return Vec::new();
    };
    let mut found: Vec<std::path::PathBuf> = entries
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "mccap"))
        .collect();
    found.sort();
    found
}

/// Real traffic, decoded and re-encoded — the test S06's "golden test against captured vanilla
/// traffic" actually wanted.
///
/// This runs against the committed fixtures with no configuration at all. `MC_PARITY_CAPTURE`
/// overrides the set with a single file; if neither exists the test skips, which is how it
/// behaved before the first capture was committed.
#[test]
fn a_real_capture_replays_clean_when_one_exists() {
    let captures = real_captures();
    if captures.is_empty() {
        eprintln!(
            "skip: no real capture available. Commit one to tests/fixtures/, or set \
             MC_PARITY_CAPTURE=<path to a .mccap>."
        );
        return;
    }

    for path in &captures {
        let name = path.file_name().unwrap_or(path.as_os_str());
        let capture = Capture::read_file(path).expect("read the real capture");
        assert_eq!(
            capture.meta.protocol,
            mc_protocol::PROTOCOL_VERSION,
            "{name:?}: a capture of a different protocol cannot be replayed by these decoders"
        );
        let report = Replayer::new(&capture).run();
        println!("real capture {name:?}: {}", report.summary());

        assert!(
            report.decode_failures.is_empty(),
            "{name:?}: frames a real server sent did not decode:\n{:#?}",
            report.decode_failures
        );
        assert!(
            report.reencode_mismatches.is_empty(),
            "{name:?}: frames decoded but did not re-encode to the server's own bytes:\n{:#?}",
            report.reencode_mismatches
        );
        assert!(report.is_clean(), "{name:?}: {}", report.summary());

        // Floors, so that "clean" cannot come to mean "clean and empty" if a fixture is ever
        // replaced by a shorter one. The two committed recordings are 898 and 1679 frames
        // across 37 and 38 distinct (state, direction, id) triples respectively.
        assert!(
            report.frames > 300,
            "{name:?}: a real session recording should be substantial, got {} frames",
            report.frames
        );
        assert!(
            report.distinct_packets() >= 20,
            "{name:?}: a real session should exercise many packet types, got {}",
            report.distinct_packets()
        );
        assert!(
            capture.frames_in(Direction::Clientbound).count() > 100
                && capture.frames_in(Direction::Serverbound).count() > 100,
            "{name:?}: both directions must be represented"
        );
    }
}
