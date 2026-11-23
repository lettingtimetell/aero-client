//! Checks the §12 hazard-coverage map's citations mechanically.
//!
//! [`mc_parity::hazards`] claims, for each of H1–H13, which file and which test carries the
//! proof. Those claims are the kind that rot: a section renames a test, another moves a module,
//! and the map silently becomes a list of pleasant assertions about code that no longer exists.
//!
//! This file makes that impossible. Every cited path must exist relative to the workspace root
//! and must contain the cited symbol. It deliberately does **not** re-run the cited tests —
//! `cargo test --workspace` does that, and a harness that re-implemented thirteen other crates'
//! assertions would just be a second place for them to be wrong.

use mc_parity::hazards::{self, ProofKind, HAZARDS};

/// Java reference generators are not part of the published source tree (no `.java` file is).
/// When a cited generator is absent, its citation is checked through the `generate.ps1` beside
/// it, which compiles and runs the generator by name; where the Java file is present it is read
/// and checked like any other proof.
fn generator_script_names(root: &std::path::Path, cited: &str) -> bool {
    let cited = std::path::Path::new(cited);
    if cited.extension().and_then(|e| e.to_str()) != Some("java") {
        return false;
    }
    let (Some(dir), Some(name)) = (cited.parent(), cited.file_stem().and_then(|s| s.to_str()))
    else {
        return false;
    };
    std::fs::read_to_string(root.join(dir).join("generate.ps1"))
        .is_ok_and(|script| script.contains(name))
}

#[test]
fn every_cited_proof_exists_and_contains_what_it_claims() {
    let root = hazards::workspace_root();
    let mut missing = Vec::new();

    for hazard in &HAZARDS {
        for proof in hazard.proofs {
            let path = root.join(proof.path);
            if !path.is_file() {
                if generator_script_names(&root, proof.path) {
                    continue;
                }
                missing.push(format!("{}: {} does not exist", hazard.id, proof.path));
                continue;
            }
            if proof.symbol.is_empty() {
                continue;
            }
            match std::fs::read_to_string(&path) {
                Ok(text) => {
                    if !text.contains(proof.symbol) {
                        missing.push(format!(
                            "{}: {} no longer contains `{}`",
                            hazard.id, proof.path, proof.symbol
                        ));
                    }
                }
                Err(e) => missing.push(format!(
                    "{}: {} could not be read ({e}) — a binary fixture should cite an empty \
                     symbol",
                    hazard.id, proof.path
                )),
            }
        }
    }

    assert!(
        missing.is_empty(),
        "the hazard map has drifted from the code:\n  {}",
        missing.join("\n  ")
    );
}

#[test]
fn the_map_is_honest_about_what_is_not_proven() {
    let unproven: Vec<&str> = HAZARDS
        .iter()
        .filter(|h| !h.is_proven())
        .map(|h| h.id)
        .collect();
    assert_eq!(
        unproven,
        vec!["H13"],
        "if this set changed, the section report has to change with it"
    );

    // A hazard whose strongest evidence is only self-consistent must never be described as
    // settled. The `NOT PROVEN` marker in the note is what the report quotes.
    for h in HAZARDS.iter().filter(|h| h.kind == ProofKind::Unproven) {
        assert!(
            h.note.contains("NOT PROVEN"),
            "{} is Unproven but its note does not say so",
            h.id
        );
    }
}

#[test]
fn four_hazards_are_backed_by_the_real_java_or_by_real_vanilla_data() {
    // H1/H5 (mc-math) and H7 (mc-physics) run the real 1.8.9 bytecode under a JDK; H2, H9 and
    // H12 are checked against files Mojang shipped. That is the strongest tier of evidence in
    // this workspace and it should not quietly shrink.
    let java: Vec<&str> = HAZARDS
        .iter()
        .filter(|h| h.kind == ProofKind::JavaDifferential)
        .map(|h| h.id)
        .collect();
    assert_eq!(java, vec!["H1", "H5", "H7"]);

    let vanilla: Vec<&str> = HAZARDS
        .iter()
        .filter(|h| h.kind == ProofKind::VanillaData)
        .map(|h| h.id)
        .collect();
    assert_eq!(vanilla, vec!["H2", "H9", "H12"]);
}

#[test]
fn print_the_map() {
    // Not an assertion — the map is most useful when a CI log carries it, so a reader can see
    // the evidence tier of every hazard without opening the source.
    println!("{}", hazards::markdown_table());
    for h in &HAZARDS {
        println!(
            "{} [{}] {}\n    {}",
            h.id,
            h.kind.label(),
            h.summary,
            h.note
        );
    }
}

#[test]
fn the_compatibility_matrix_runbook_requires_live_evidence() {
    let book = mc_parity::matrix::runbook();
    assert!(book.contains(mc_parity::matrix::EXECUTION_REQUIREMENTS));
    println!("{book}");
    println!("{}", mc_parity::matrix::EXECUTION_REQUIREMENTS);
}
