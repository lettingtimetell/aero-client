//! Compatibility and ordinary-gameplay anticheat acceptance criteria.
//!
//! This module describes checks; it does not discover servers or execute them. Live execution
//! is opt-in through client integration tests. Keep version/configuration receipts, packet
//! captures, replay reports, and server logs with each run. A skipped test is not a pass.
//! Current execution evidence is indexed in parity/COMPLETION.md. Offline differential
//! physics tests support these checks but do not replace live observations.

/// What kind of server implementation a target is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServerKind {
    /// Mojang's own server jar. The definition of correct.
    Vanilla,
    /// A Bukkit-family fork: vanilla plus a plugin API and its own patches.
    Bukkit,
    /// A proxy that sits between the client and a backend server.
    Proxy,
}

/// One server the matrix would be run against.
#[derive(Debug, Clone, Copy)]
pub struct ServerTarget {
    /// Display name.
    pub name: &'static str,
    /// What kind of implementation it is.
    pub kind: ServerKind,
    /// The build to use.
    pub version: &'static str,
    /// Official source or build procedure.
    pub source: &'static str,
    /// What this target specifically risks that the others do not — the reason it earns a row.
    pub distinct_risk: &'static str,
}

/// The four targets §7 names.
pub const TARGETS: [ServerTarget; 4] = [
    ServerTarget {
        name: "Vanilla",
        kind: ServerKind::Vanilla,
        version: "1.8.9 (server.jar, protocol 47)",
        source: "Mojang version manifest server download.",
        distinct_risk: "The baseline. Anything that fails here is this client's bug, full stop, \
                        with no third-party implementation to blame.",
    },
    ServerTarget {
        name: "Spigot",
        kind: ServerKind::Bukkit,
        version: "1.8.8 / 1.8.9, built with BuildTools",
        source: "SpigotMC BuildTools; retain the build and configuration receipt.",
        distinct_risk: "Anti-xray rewrites chunk data before it is sent, and Spigot's entity \
                        tracker batches movement differently. Both land squarely on S08's \
                        chunk unpacking and S14's entity sync.",
    },
    ServerTarget {
        name: "Paper",
        kind: ServerKind::Bukkit,
        version: "Paper 1.8.8 (the last 1.8 line)",
        source: "PaperMC downloads API.",
        distinct_risk: "Paper rewrote the chunk sender and the packet compression path, and is \
                        far more aggressive about timing out a client that stalls. It is the \
                        best available test of the inbound backpressure design in \
                        `mc-client`'s `net` module.",
    },
    ServerTarget {
        name: "BungeeCord",
        kind: ServerKind::Proxy,
        version: "BungeeCord #1.8-compatible build, in front of two vanilla backends",
        source: "SpigotMC's Jenkins.",
        distinct_risk: "The proxy terminates encryption itself and re-sends `LoginSuccess` with \
                        an offline-mode UUID; a *server switch* replays `JoinGame` mid-session, \
                        which no single-server test ever exercises. If the client assumes \
                        `JoinGame` arrives once, this is where it breaks.",
    },
];

/// One thing the matrix checks, on every target.
#[derive(Debug, Clone, Copy)]
pub struct MatrixCheck {
    /// Short id, for the results table.
    pub id: &'static str,
    /// What to do.
    pub action: &'static str,
    /// What must be true afterwards — written so it is checkable, not "it works".
    pub assertion: &'static str,
    /// What to keep from the run, so a failure is diagnosable later.
    pub evidence: &'static str,
}

/// The checks, in the order a run should do them: cheapest and most-blocking first.
pub const CHECKS: [MatrixCheck; 8] = [
    MatrixCheck {
        id: "M1",
        action: "Server-list ping (`Handshake` next=1, `Request`, `Ping`).",
        assertion: "A `Response` decodes, its JSON has `version.protocol == 47`, and the `Pong` \
                    echoes the sent payload exactly.",
        evidence: "The status JSON, verbatim.",
    },
    MatrixCheck {
        id: "M2",
        action: "Offline-mode login to the Play state.",
        assertion: "`LoginSuccess` decodes, the UUID parses, and the connection reaches Play \
                    without a `Disconnect`.",
        evidence: "A `mc-parity` capture of the whole login, from the first byte.",
    },
    MatrixCheck {
        id: "M3",
        action: "Online-mode login (vanilla and Paper; BungeeCord terminates it itself).",
        assertion: "`EncryptionRequest` → session join → `EncryptionResponse` is accepted, and \
                    the first packet *after* the cipher engages decodes — hazard H4's failure \
                    signature is precisely one good packet followed by garbage.",
        evidence: "A capture, plus the frame index at which decoding first failed if it did.",
    },
    MatrixCheck {
        id: "M4",
        action: "Join and stand still for 60 seconds.",
        assertion: "No disconnect; every `KeepAlive` is answered; `mc-parity` replay of the \
                    capture reports zero decode failures and zero unknown ids.",
        evidence: "The capture and its `ReplayReport`.",
    },
    MatrixCheck {
        id: "M5",
        action: "Walk a 200-block loop through loaded terrain.",
        assertion: "No `PlayerPositionLook` correction is received other than the initial join \
                    teleport. A correction mid-walk means client and server physics diverged, \
                    which is H7 showing up on the wire.",
        evidence: "Every `PlayerPositionLook` frame, with its tick index.",
    },
    MatrixCheck {
        id: "M6",
        action: "Break and place one block; open a chest, drag-split a stack, close it.",
        assertion: "The client's predicted container state matches the server's `WindowItems` \
                    after the transaction, with no `ConfirmTransaction` rejection. This is \
                    hazard H10's lag-only ghost item, which only a real server can produce.",
        evidence: "The capture, and a diff of predicted vs received slot contents.",
    },
    MatrixCheck {
        id: "M7",
        action: "Replay every capture taken above through `mc_parity::Replayer`.",
        assertion: "`ReplayReport::is_clean()` — every frame decodes *and* re-encodes \
                    byte-identically. This is the check that turns a session into a permanent \
                    regression fixture.",
        evidence: "The report, and the captures committed as fixtures.",
    },
    MatrixCheck {
        id: "M8",
        action: "BungeeCord only: switch backend servers mid-session.",
        assertion: "The backend boundary (`JoinGame` or `Respawn`) is handled with the \
                    corresponding world reset, terrain reload, and entity lifecycle.",
        evidence: "A capture spanning the switch.",
    },
];

/// The anticheat sanity run, which is a different and harder thing from the matrix.
pub const ANTICHEAT_STEPS: [MatrixCheck; 4] = [
    MatrixCheck {
        id: "A1",
        action: "Join an authorized 1.8.9 server with active anticheat checks and accessible server flag logs; an isolated local fixture may use offline authentication.",
        assertion: "No kick or server-side flag during 5 minutes of idling, using a non-operator account without bypass permissions.",
        evidence: "Capture, plugin version/configuration, account permissions, and server logs.",
    },
    MatrixCheck {
        id: "A2",
        action: "Walk, sprint, sprint-jump, and stair-climb for 5 minutes.",
        assertion: "No `Speed`, `NoFall` or `Fly` flag. Sprint-jumping is the classic false \
                    positive and is exactly what H1's LUT and H7's Y→X→Z ordering exist to get \
                    right.",
        evidence: "A capture; the C03 shape histogram from `mc-parity`'s replay.",
    },
    MatrixCheck {
        id: "A3",
        action: "Fall 20 blocks into water, and separately onto land.",
        assertion: "No `NoFall` flag; the damage taken matches the client's own \
                    `predicted_fall_damage`, which is bit-exact against Java offline.",
        evidence: "A capture and the predicted-vs-actual damage.",
    },
    MatrixCheck {
        id: "A4",
        action: "Mine a 30-block tunnel at survival speed.",
        assertion: "No `FastBreak` or `BadPackets` flag; every `PlayerDigging` start is followed \
                    by exactly one stop at the right tick.",
        evidence: "A capture.",
    },
];

/// Requirements for live evidence, independent of the host running this module.
pub const EXECUTION_REQUIREMENTS: &str = "Live checks require explicit target configuration, compatible server binaries/JVMs, and retained captures with clean replay reports. Online login also requires launcher-supplied entitled-account credentials. Anticheat checks require active checks, a non-operator account without bypass permissions, and server flag logs. Consult parity/COMPLETION.md for measured results; this runbook is not a result.";

/// Render the whole runbook as Markdown.
#[must_use]
pub fn runbook() -> String {
    let mut out = String::new();
    out.push_str("# Multi-server compatibility matrix (runbook)\n\n");
    out.push_str(EXECUTION_REQUIREMENTS);
    out.push_str(
        "\n\n## Targets\n\n| Server | Kind | Build | Distinct risk |\n|---|---|---|---|\n",
    );
    for t in &TARGETS {
        out.push_str(&format!(
            "| {} | {:?} | {} | {} |\n",
            t.name, t.kind, t.version, t.distinct_risk
        ));
    }
    out.push_str(
        "\n## Checks, run against every target\n\n| # | Action | Assertion |\n|---|---|---|\n",
    );
    for c in &CHECKS {
        out.push_str(&format!("| {} | {} | {} |\n", c.id, c.action, c.assertion));
    }
    out.push_str(
        "\n# Anticheat sanity run (runbook)\n\n| # | Action | Assertion |\n|---|---|---|\n",
    );
    for c in &ANTICHEAT_STEPS {
        out.push_str(&format!("| {} | {} | {} |\n", c.id, c.action, c.assertion));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_target_and_check_is_fully_specified() {
        assert_eq!(
            TARGETS.len(),
            4,
            "S7 names vanilla, Spigot, Paper, BungeeCord"
        );
        for t in &TARGETS {
            assert!(
                !t.source.is_empty() && !t.distinct_risk.is_empty(),
                "{}",
                t.name
            );
        }
        for c in CHECKS.iter().chain(ANTICHEAT_STEPS.iter()) {
            assert!(!c.action.is_empty() && !c.assertion.is_empty() && !c.evidence.is_empty());
            assert!(
                !c.assertion.to_lowercase().contains("it works"),
                "{} has an unfalsifiable assertion",
                c.id
            );
        }
    }

    #[test]
    fn the_runbook_mentions_every_target_and_every_check() {
        let book = runbook();
        assert!(book.contains(EXECUTION_REQUIREMENTS));
        for t in &TARGETS {
            assert!(book.contains(t.name), "{} missing", t.name);
        }
        for c in CHECKS.iter().chain(ANTICHEAT_STEPS.iter()) {
            assert!(book.contains(c.id), "{} missing", c.id);
        }
    }
}
