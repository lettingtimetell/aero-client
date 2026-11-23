//! Regression replay of the accepted 2026-09-10 live matrix captures.
//!
//! The manifest is deliberately data-driven: it binds each retained capture to the server,
//! mode, endpoint, gate, and SHA-256 recorded when that gate passed.  This test checks the
//! committed bytes decode and re-encode cleanly on every future mc-parity test run.

use std::{collections::BTreeSet, path::Path};

use mc_parity::{capture::Capture, Direction, Replayer};
use mc_protocol::State;
use serde_json::Value;
use sha2::{Digest, Sha256};

const MANIFEST: &str = include_str!("fixtures/live-matrix-2026-09-10/manifest.json");
const M1_STATUS: &str = include_str!("fixtures/live-matrix-2026-09-10/m1-status.json");
const TARGETS: [&str; 4] = ["Vanilla", "Paper", "BungeeCord", "Spigot"];

fn target(server: &str) -> &'static str {
    TARGETS
        .into_iter()
        .find(|target| server.starts_with(target))
        .unwrap_or_else(|| panic!("unknown matrix server: {server}"))
}

#[test]
fn accepted_live_matrix_captures_replay_clean() {
    let manifest: Value = serde_json::from_str(MANIFEST).expect("live matrix manifest JSON");
    assert_eq!(manifest["schema"].as_u64(), Some(1));
    assert_eq!(manifest["protocol"].as_i64(), Some(47));
    let entries = manifest["captures"]
        .as_array()
        .expect("live matrix captures array");
    let status: Value = serde_json::from_str(M1_STATUS).expect("M1 status receipt JSON");
    assert_eq!(status["schema"].as_u64(), Some(1));
    assert_eq!(status["protocol"].as_i64(), Some(47));
    assert_eq!(status["checks"]["status_json_preserved_verbatim"], true);
    assert_eq!(status["checks"]["ping_pong_payload_echoed"], true);
    let status_receipts = status["receipts"].as_array().expect("M1 receipts array");
    assert_eq!(
        status_receipts.len(),
        4,
        "M1 must cover four server targets"
    );
    let mut status_targets = BTreeSet::new();
    for receipt in status_receipts {
        status_targets.insert(target(receipt["server"].as_str().expect("M1 server")));
        let raw = receipt["status_json"]
            .as_str()
            .expect("M1 raw status JSON string");
        let parsed: Value = serde_json::from_str(raw).expect("M1 raw status JSON");
        assert_eq!(parsed["version"]["protocol"].as_i64(), Some(47));
    }
    assert_eq!(status_targets, BTreeSet::from(TARGETS));

    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/live-matrix-2026-09-10");
    let mut coverage = BTreeSet::new();
    let mut files = BTreeSet::new();
    for entry in entries {
        let file = entry["file"].as_str().expect("capture file");
        assert!(files.insert(file), "duplicate capture: {file}");
        let sha256 = entry["sha256"].as_str().expect("capture SHA-256");
        assert_eq!(
            sha256.len(),
            64,
            "{file}: manifest SHA-256 must be complete"
        );
        assert!(
            entry["server"].as_str().is_some(),
            "{file}: server provenance"
        );
        assert!(entry["mode"].as_str().is_some(), "{file}: mode provenance");
        assert!(
            entry["endpoint"].as_str().is_some(),
            "{file}: endpoint provenance"
        );
        let gate = entry["gate"].as_str().expect("gate provenance");
        let server = target(entry["server"].as_str().expect("server provenance"));
        coverage.insert((server, if gate == "S21" { "M3" } else { gate }));
        if matches!(gate, "M3" | "S21") {
            assert_eq!(entry["mode"], "online", "{file}: online login evidence");
        }
        let expected_frames = entry["frames"].as_u64().expect("frame count provenance") as usize;
        let path = root.join(file);
        let bytes = std::fs::read(&path)
            .unwrap_or_else(|error| panic!("{file}: read retained bytes {path:?}: {error:?}"));
        let actual_sha256 = format!("{:X}", Sha256::digest(&bytes));
        assert_eq!(actual_sha256, sha256, "{file}: manifest hash mismatch");
        let capture = Capture::parse(&bytes)
            .unwrap_or_else(|error| panic!("{file}: read retained capture {path:?}: {error:?}"));
        assert_eq!(capture.meta.protocol, 47, "{file}: protocol");
        assert_eq!(
            capture.len(),
            expected_frames,
            "{file}: manifest frame count"
        );
        let clientbound = capture.frames_in(Direction::Clientbound).count();
        let serverbound = capture.frames_in(Direction::Serverbound).count();
        assert!(
            clientbound > 0 && serverbound > 0,
            "{file}: accepted evidence must contain both directions"
        );
        if gate == "M8" {
            assert_eq!(server, "BungeeCord", "{file}: proxy boundary evidence");
            let ids: BTreeSet<i32> = capture
                .frames_in(Direction::Clientbound)
                .filter(|frame| frame.state == State::Play)
                .filter_map(|frame| frame.packet_id())
                .collect();
            assert!(ids.contains(&1), "{file}: M8 must contain JoinGame");
            assert!(
                ids.contains(&7),
                "{file}: M8 must contain the Respawn boundary"
            );
        }
        let report = Replayer::new(&capture).run();
        assert!(
            report.is_clean(),
            "{file}: retained capture is no longer clean: {report:?}"
        );
        assert_eq!(report.byte_identical, expected_frames);
        println!("{server} {gate}: {file}: {}", report.summary());
    }
    for server in TARGETS {
        for gate in ["M3", "M4", "M5", "M6"] {
            assert!(
                coverage.contains(&(server, gate)),
                "missing {server} {gate}"
            );
        }
    }
    assert!(coverage.contains(&("BungeeCord", "M8")));
}
