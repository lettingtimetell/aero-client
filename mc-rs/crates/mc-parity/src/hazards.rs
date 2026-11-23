//! The §12 hazard table as machine-checked data.
//!
//! There are thirteen hazards — the specific, known ways a 1.8.9 reimplementation
//! goes subtly wrong. Thirteen sections each dealt with one or two of them, in thirteen
//! different crates, and the evidence is spread across the workspace. That is fine right up
//! until someone asks "is H4 actually proven, or does everyone just believe it is?", at which
//! point the answer takes an afternoon of grepping.
//!
//! This module is that answer, written down: for each hazard, **what kind of reference the
//! proof is measured against**, and **which file and which test name** carries it.
//! `tests/hazards.rs` then checks the citations mechanically — every cited file must exist and
//! must contain the cited symbol — so the map cannot rot into a list of pleasant claims about
//! tests that were renamed or deleted two sections ago.
//!
//! It deliberately does **not** re-run those tests. `cargo test --workspace` already does that,
//! and a harness that re-implemented thirteen other crates' assertions would be a second place
//! for them to be wrong.
//!
//! # How to read [`ProofKind`]
//!
//! The kinds describe **how independent the reference is from the code being checked**, which
//! is the only thing that makes a passing test evidence rather than a tautology:
//!
//! * [`ProofKind::JavaDifferential`] — measured against output from the *real* recompiled 1.8.9
//!   bytecode running under a JVM. Nothing is stronger; a passing test here means the Rust and
//!   the Java agree, not that the Rust agrees with itself.
//! * [`ProofKind::VanillaData`] — measured against a real file Mojang shipped (`1.8.9.jar`,
//!   `servers.dat`, `glyph_sizes.bin`).
//! * [`ProofKind::PublicVector`] — measured against a published test vector from outside this
//!   project entirely (NIST, a standards paper, the protocol's documented examples).
//! * [`ProofKind::HandDerived`] — measured against numbers a human transcribed from the Java
//!   source or the protocol spec, by a route that does not run the implementation.
//! * [`ProofKind::SelfConsistent`] — round-trip or internal-consistency only. Real evidence
//!   about robustness; **no** evidence about parity.
//! * [`ProofKind::Unproven`] — no mechanical check exists in this workspace.
//!
//! [`Hazard::is_proven`] is true for the first four and false for the last two. Two hazards are
//! not proven, and both say so in their `note` rather than being quietly upgraded.

/// How independent a hazard's reference is from the code it checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProofKind {
    /// Compared against the real recompiled 1.8.9 bytecode, run under a JVM.
    JavaDifferential,
    /// Compared against a real file Mojang shipped.
    VanillaData,
    /// Compared against a published test vector from outside this project.
    PublicVector,
    /// Compared against numbers transcribed by hand from the Java source or the wire spec.
    HandDerived,
    /// Round-trip or internal consistency only — robustness evidence, not parity evidence.
    SelfConsistent,
    /// No mechanical check exists here.
    Unproven,
}

impl ProofKind {
    /// A short label for a printed table.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            ProofKind::JavaDifferential => "java-differential",
            ProofKind::VanillaData => "vanilla-data",
            ProofKind::PublicVector => "public-vector",
            ProofKind::HandDerived => "hand-derived",
            ProofKind::SelfConsistent => "self-consistent",
            ProofKind::Unproven => "UNPROVEN",
        }
    }
}

/// One citation: a workspace-relative file and a symbol that must appear in it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HazardProof {
    /// Path relative to the `mc-rs` workspace root.
    pub path: &'static str,
    /// A test function name, or another symbol, that must be present in that file.
    pub symbol: &'static str,
}

/// One row of §12, with its evidence.
#[derive(Debug, Clone, Copy)]
pub struct Hazard {
    /// `"H1"` … `"H13"`.
    pub id: &'static str,
    /// The hazard, restated in one line.
    pub summary: &'static str,
    /// The plan sections it bites.
    pub sections: &'static str,
    /// How independent the strongest available evidence is.
    pub kind: ProofKind,
    /// Where that evidence lives.
    pub proofs: &'static [HazardProof],
    /// What the evidence does and does not settle.
    pub note: &'static str,
}

impl Hazard {
    /// Whether an independent mechanical check exists for this hazard.
    #[must_use]
    pub const fn is_proven(&self) -> bool {
        matches!(
            self.kind,
            ProofKind::JavaDifferential
                | ProofKind::VanillaData
                | ProofKind::PublicVector
                | ProofKind::HandDerived
        )
    }
}

/// The thirteen hazards of §12, with their evidence.
pub const HAZARDS: [Hazard; 13] = [
    Hazard {
        id: "H1",
        summary: "MathHelper sin/cos is a 65536-entry LUT, not libm; f32::sin desyncs movement \
                  over hundreds of ticks.",
        sections: "S05, S13",
        kind: ProofKind::JavaDifferential,
        proofs: &[
            HazardProof {
                path: "crates/mc-math/javaref/MathRefGen.java",
                symbol: "MathHelper",
            },
            HazardProof {
                path: "crates/mc-math/tests/differential.rs",
                symbol: "sin_table_is_the_jvm_table",
            },
            HazardProof {
                path: "crates/mc-math/tests/differential.rs",
                symbol: "recomputing_the_table_locally_is_not_a_substitute",
            },
        ],
        note: "The whole 65536-entry table is compared with the JVM's, entry by entry, on raw \
               IEEE-754 bits with zero tolerance. The second test is the interesting one: it \
               asserts that rebuilding the table from the same formula in Rust does *not* \
               reproduce it, so the fixture is load-bearing rather than decorative.",
    },
    Hazard {
        id: "H2",
        summary: "Java Modified UTF-8 in NBT: NUL becomes two bytes, supplementary characters \
                  become CESU-8 surrogate pairs.",
        sections: "S02",
        kind: ProofKind::VanillaData,
        proofs: &[
            HazardProof {
                path: "crates/mc-nbt/src/modified_utf8.rs",
                symbol: "nul_is_two_bytes",
            },
            HazardProof {
                path: "crates/mc-nbt/src/modified_utf8.rs",
                // A lone surrogate now decodes to a private-use escape
                // and re-encodes byte for byte, as Java's reader accepts it.
                symbol: "a_lone_surrogate_decodes_to_its_escape_and_round_trips",
            },
            HazardProof {
                path: "crates/mc-nbt/tests/golden.rs",
                symbol: "servers.dat",
            },
            HazardProof {
                path: "crates/mc-parity/src/fuzz.rs",
                symbol: "TRICKY_STRINGS",
            },
        ],
        note: "The encoder and decoder are pinned on the two divergent cases, and the real \
               vanilla `jars/servers.dat` round-trips byte-identically. This crate's fuzz \
               generator additionally feeds an embedded NUL and a supplementary character into \
               every NBT round-trip it runs.",
    },
    Hazard {
        id: "H3",
        summary: "The server-join hash is SHA-1 rendered as signed twos-complement hex; a \
                  negative digest gets a `-` prefix.",
        sections: "S03, S07",
        kind: ProofKind::PublicVector,
        proofs: &[
            HazardProof {
                path: "crates/mc-auth/src/hash.rs",
                symbol: "known_test_vectors",
            },
            HazardProof {
                path: "crates/mc-auth/src/hash.rs",
                symbol: "simon_has_its_leading_zero_stripped",
            },
            HazardProof {
                path: "crates/mc-auth/src/hash.rs",
                symbol: "signed_hex_edges",
            },
        ],
        note: "Checked against the three public vectors (`Notch`, `jeb_`, `simon`), which \
               between them cover a positive digest, a negative one, and one whose leading \
               zero must be stripped. This is a closed question — the vectors come from \
               outside the project.",
    },
    Hazard {
        id: "H4",
        summary: "AES-CFB8 must be one continuous stream across packets; re-initialising per \
                  packet fails exactly one packet after login.",
        sections: "S07",
        kind: ProofKind::PublicVector,
        proofs: &[
            HazardProof {
                path: "crates/mc-protocol/src/crypt.rs",
                symbol: "reinitialising_per_packet_is_wrong_after_the_first",
            },
            HazardProof {
                path: "crates/mc-protocol/tests/s07_golden.rs",
                symbol: "cfb8_matches_a_from_scratch_implementation",
            },
        ],
        note: "The cipher itself is checked against NIST SP 800-38A F.3.7, and there is a \
               dedicated regression test for the failure signature: a per-packet-reinitialised \
               cipher matches on packet 1 and diverges on packets 2 and 3. What remains \
               unproven is only that a real server accepts the stream, which needs a server.",
    },
    Hazard {
        id: "H5",
        summary: "Protocol `Position` packs x/y/z as 26/12/26 bits in an i64, sign-extended; a \
                  wrong sign only shows up away from the origin.",
        sections: "S06",
        kind: ProofKind::JavaDifferential,
        proofs: &[
            HazardProof {
                path: "crates/mc-math/tests/differential.rs",
                symbol: "blockpos_and_facing",
            },
            HazardProof {
                path: "crates/mc-math/tests/fixtures/blockpos_from_long.bin",
                symbol: "",
            },
            HazardProof {
                path: "crates/mc-parity/src/synthetic.rs",
                symbol: "-33_554_000",
            },
        ],
        note: "`BlockPos::to_long`/`from_long` are compared against the JVM's own \
               `BlockPos.toLong`/`fromLong` over dumped fixtures, and `mc-protocol` reuses \
               those exact functions rather than re-deriving the packing. This crate's \
               synthetic session deliberately places a block near the far negative edge of the \
               26-bit range, because a capture confined to the positive octant cannot catch a \
               sign error.",
    },
    Hazard {
        id: "H6",
        summary: "Chunk section blocks are little-endian u16; sky light is absent outside the \
                  overworld, and reading it unconditionally desyncs the buffer.",
        sections: "S08",
        kind: ProofKind::HandDerived,
        proofs: &[
            HazardProof {
                path: "crates/mc-world/tests/chunk_data_fixture.rs",
                symbol: "s21_chunk_data_synthetic_fixture_decodes_to_known_states",
            },
            HazardProof {
                path: "crates/mc-world/tests/chunk_data_fixture.rs",
                symbol:
                    "nether_shaped_fixture_has_no_sky_light_region_and_biomes_still_land_correctly",
            },
            HazardProof {
                path: "crates/mc-world/tests/chunk_data_fixture.rs",
                symbol: "s26_map_chunk_bulk_decodes_multiple_columns_and_sets_dimension_sky_flag",
            },
        ],
        note: "The both-halves of this hazard are covered: byte order, and the nether-shaped \
               buffer with no sky-light region where the biome array must still land at the \
               right offset. The fixtures are hand-assembled from the documented wire layout, \
               because no real chunk packet was available when S08 ran — which is exactly the \
               gap a real capture through this crate would close.",
    },
    Hazard {
        id: "H7",
        summary: "Movement is f64 throughout, applied Y then X then Z, with step-up retried \
                  after the first sweep.",
        sections: "S13",
        kind: ProofKind::JavaDifferential,
        proofs: &[
            HazardProof {
                path: "crates/mc-physics/javaref/PhysicsRefGen.java",
                symbol: "moveEntity",
            },
            HazardProof {
                path: "crates/mc-physics/tests/parity.rs",
                symbol: "parity_is_bit_exact_not_merely_within_tolerance",
            },
            HazardProof {
                path: "crates/mc-parity/src/physics_trace.rs",
                symbol: "SCENARIOS",
            },
        ],
        note: "The strongest evidence in the project: a real `EntityPlayer` driven through real \
               `Entity.moveEntity` bytecode over 22 scenarios and 6,743 ticks, compared on raw \
               IEEE-754 bits at tolerance 0.0. S13 shipped twelve of them (4,570 ticks; its \
               handoff entry says 4,590). GAP A11 added ten more that drive the branches the \
               reference generator used to omit: creative flight, a server-sent movement speed, \
               knockback, Jump Boost, item use, Depth Strider, spectator `noClip`, a dead \
               player, an unloaded column and the cactus/portal contact callbacks. This crate \
               re-proves the same fixtures through its generic comparator, with the flag byte \
               split into seven named channels.",
    },
    Hazard {
        id: "H8",
        summary: "Vanilla's ambient-occlusion formula uses three neighbours per vertex, with a \
                  special case when both side blocks are opaque.",
        sections: "S11",
        kind: ProofKind::HandDerived,
        proofs: &[
            HazardProof {
                path: "crates/mc-render/tests/s11_ao.rs",
                symbol: "both_side_blocks_opaque_skips_the_corner_sample",
            },
            HazardProof {
                path: "crates/mc-render/tests/s11_ao.rs",
                symbol: "a_wall_on_one_side_darkens_exactly_two_corners",
            },
            HazardProof {
                path: "crates/mc-render/tests/s11_ao.rs",
                symbol: "an_occupied_diagonal_darkens_the_corner_even_with_clear_sides",
            },
        ],
        note: "Hand-derived vanilla numbers per corner, including the special case asserted \
               *positively* (0.4) and the naive always-sample-the-diagonal answer (0.6) \
               asserted not to occur. That negative assertion is what makes it a real check \
               rather than a restatement of the implementation.",
    },
    Hazard {
        id: "H9",
        summary: "Font glyph widths come from `glyph_sizes.bin`, not from measuring the texture.",
        sections: "S15",
        kind: ProofKind::VanillaData,
        proofs: &[
            HazardProof {
                path: "crates/mc-ui/src/font/glyphs.rs",
                symbol: "GLYPH_SIZES",
            },
            HazardProof {
                path: "crates/mc-ui/src/font/tests.rs",
                symbol: "unicode_widths_come_from_the_byte_table_not_the_texture",
            },
            HazardProof {
                path: "crates/mc-ui/src/font/tests.rs",
                symbol: "the_cursor_advance_truncates_where_the_width_table_rounds",
            },
        ],
        note: "The test name is the assertion: widths come from the byte table. The sign \
               extension of the 50 high-bit bytes in the real `glyph_sizes.bin` is handled and \
               documented at the source.",
    },
    Hazard {
        id: "H10",
        summary: "Container drag-click has three protocol phases; approximations create \
                  lag-only ghost items.",
        sections: "S17",
        kind: ProofKind::HandDerived,
        proofs: &[
            HazardProof {
                path: "crates/mc-ui/src/screens/container/interaction.rs",
                symbol: "a_full_drag_lifecycle_produces_start_add_add_end_in_order",
            },
            HazardProof {
                path: "crates/mc-ui/src/screens/container/interaction.rs",
                symbol: "feeding_the_drag_events_into_the_real_container_actually_splits_the_stack",
            },
            HazardProof {
                path: "crates/mc-ui/src/screens/container/model.rs",
                symbol: "even_split_drag_with_a_remainder_keeps_the_remainder_on_the_cursor",
            },
        ],
        note: "All three phases are emitted in order and the local container model agrees with \
               what the phases say. The part this cannot reach is the *lag-only* symptom, which \
               by definition needs a server with latency: ghost items appear when the client's \
               prediction and the server's `Container.slotClick` disagree, and there is no \
               server here to disagree with.",
    },
    Hazard {
        id: "H11",
        summary: "Movement packet cadence is anticheat-relevant independently of position \
                  correctness.",
        sections: "S14",
        kind: ProofKind::HandDerived,
        proofs: &[
            HazardProof {
                path: "crates/mc-physics/src/prediction.rs",
                symbol:
                    "a_standing_player_sends_the_bare_ground_packet_except_every_twentieth_tick",
            },
            HazardProof {
                path: "crates/mc-physics/src/prediction.rs",
                symbol: "walking_sends_position_packets",
            },
            HazardProof {
                path: "crates/mc-physics/src/prediction.rs",
                symbol: "turning_on_the_spot_sends_a_look_packet",
            },
        ],
        note: "`Predictor::outbound` reproduces `onUpdateWalkingPlayer`'s choice between the \
               four C03 shapes, including the 9.0E-4 squared-distance threshold and the forced \
               full update every 20 ticks, and the *positions* the choice is made from are \
               bit-exact against real Java (H7). The cadence rule itself is transcribed from \
               the Java rather than differentially tested, because `PhysicsRefGen` dumps \
               positions, not packet choices. Nothing here has been shown to a live anticheat.",
    },
    Hazard {
        id: "H12",
        summary: "Model `uvlock` plus parent-chain `#texture` indirection — stairs and fences \
                  are the canaries.",
        sections: "S09",
        kind: ProofKind::VanillaData,
        proofs: &[
            HazardProof {
                path: "crates/mc-assets/src/model/facebakery.rs",
                symbol: "uvlock_keeps_texture_aligned_to_world_space",
            },
            HazardProof {
                path: "crates/mc-assets/src/model/blockstate.rs",
                symbol: "stairs_rotation_and_uvlock",
            },
            HazardProof {
                path: "crates/mc-assets/tests/s09_model_bake.rs",
                symbol: "stairs_bake_with_geometry_in_every_orientation",
            },
            HazardProof {
                path: "crates/mc-assets/tests/s09_model_bake.rs",
                symbol: "atlas_region_is_a_byte_exact_copy_of_the_source_png",
            },
        ],
        note: "Both named canaries bake with geometry in every orientation, straight out of the \
               real `1.8.9.jar`, and a stitched atlas region is byte-identical to the source \
               PNG. What is checked is that the geometry and the atlas are right, not that a \
               rendered stair *looks* like vanilla's — that needs the vanilla screenshot this \
               environment does not have (see `crate::snapshot`).",
    },
    Hazard {
        id: "H13",
        summary: "Yggdrasil auth is retired; only the Microsoft flow works, and it needs your \
                  own Azure app registration.",
        sections: "S03",
        kind: ProofKind::Unproven,
        proofs: &[
            HazardProof {
                path: "crates/mc-auth/tests/flow.rs",
                symbol: "full_device_code_login",
            },
            HazardProof {
                path: "crates/mc-auth/tests/flow.rs",
                symbol: "login_silently_refreshes_an_expired_cached_token",
            },
        ],
        note: "NOT PROVEN, and not provable here. The whole MSA → XBL → XSTS → Minecraft chain \
               is implemented and thoroughly tested, but every test is against `wiremock`, \
               never against the live services — which makes the evidence self-consistent, not \
               independent. Closing this needs an Azure app registration and an interactive \
               human login, which is S03's own unticked `Live verification` task. The two \
               citations are listed as supporting evidence, not as a proof.",
    },
];

/// The `mc-rs` workspace root, which every [`HazardProof::path`] is relative to.
#[must_use]
pub fn workspace_root() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
}

/// Render the map as a Markdown table, for pasting into a report.
#[must_use]
pub fn markdown_table() -> String {
    let mut out = String::from("| # | Proof kind | Proven? | Where |\n|---|---|---|---|\n");
    for h in &HAZARDS {
        let where_ = h
            .proofs
            .iter()
            .map(|p| {
                if p.symbol.is_empty() {
                    p.path.to_owned()
                } else {
                    format!("`{}`::{}", p.path, p.symbol)
                }
            })
            .collect::<Vec<_>>()
            .join("<br>");
        out.push_str(&format!(
            "| {} | {} | {} | {} |\n",
            h.id,
            h.kind.label(),
            if h.is_proven() { "yes" } else { "**no**" },
            where_
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_map_covers_h1_to_h13_in_order() {
        assert_eq!(HAZARDS.len(), 13);
        for (i, h) in HAZARDS.iter().enumerate() {
            assert_eq!(h.id, format!("H{}", i + 1), "row {i} is out of order");
            assert!(!h.summary.is_empty());
            assert!(!h.note.is_empty(), "{} has no note", h.id);
            assert!(!h.proofs.is_empty(), "{} cites nothing", h.id);
        }
    }

    #[test]
    fn exactly_the_hazards_we_say_are_unproven_are_unproven() {
        // Written as an exact set rather than a count, so an accidental upgrade of a hazard's
        // `kind` fails here instead of quietly improving the project's self-assessment.
        let unproven: Vec<&str> = HAZARDS
            .iter()
            .filter(|h| !h.is_proven())
            .map(|h| h.id)
            .collect();
        assert_eq!(
            unproven,
            vec!["H13"],
            "the set of unproven hazards changed; if that is intentional, say so in the report"
        );
    }

    #[test]
    fn a_note_that_claims_a_hazard_is_unproven_agrees_with_its_kind() {
        for h in &HAZARDS {
            let claims_unproven = h.note.contains("NOT PROVEN");
            assert_eq!(
                claims_unproven,
                h.kind == ProofKind::Unproven,
                "{}'s note and its ProofKind disagree",
                h.id
            );
        }
    }

    #[test]
    fn the_markdown_table_names_every_hazard() {
        let table = markdown_table();
        for h in &HAZARDS {
            assert!(table.contains(h.id), "{} missing from the table", h.id);
        }
    }
}
