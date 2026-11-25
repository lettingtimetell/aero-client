#![forbid(unsafe_code)]
#![warn(missing_docs)]

//! The Minecraft version the client plays.
//!
//! The client is a Minecraft 1.8.9 client (protocol 47) and can also play 1.7.10 (protocol 5),
//! chosen with the version slider on the title screen. 1.8.9 is the default, and choosing it
//! leaves every behavior exactly as it is without this choice.
//!
//! This crate holds that choice and nothing else, so the protocol, the game logic, the settings
//! and the screens can all name it without depending on one another.
//!
//! The choice is read once when a connection opens and then stays fixed for that connection:
//! code inside a session asks the session for its version and never re-reads the setting.
//!
//! # Names used across the workspace
//!
//! - **`v5`, `_v5`, `V5`** mark 1.7.10 code, after its protocol number. A `v5` module
//!   (`mc_protocol::v5`, `mc_data::v5`, `mc_physics::v5`, `mc_world::v5`) or a `*_v5.rs` file
//!   holds 1.7.10's half of what its neighbour does for 1.8.9, and a `_v5` function or `V5` type
//!   is the 1.7.10 counterpart of a 1.8.9 one (`PlayerV5`). The protocol-5 packet enums are
//!   `S2CPlay5` and `C2SPlay5`. 1.8.9 code keeps its plain names; where both versions meet, the
//!   1.8.9 side may say `v47` (`C2SPlay5::from_v47`).
//! - **`v17_`** marks the differential tests that run the same scenarios on the real 1.7.10
//!   client, with their fixtures and generators (`v17_movement_oracle`,
//!   `tests/fixtures/v17_oracle`, `generate-v17-oracle.ps1`). A test that checks both versions
//!   against their Java keeps a fixture per version (`crafting_1_7_10.txt`, `crafting_1_8_9.txt`).
//! - **`java17:` and `java18:`** in comments cite vanilla 1.7.10's and 1.8.9's source: the class
//!   and method, often with the file and line. The 1.8.9 code written before 1.7.10 support
//!   writes a plain `java:` for 1.8.9; elsewhere a plain `java:` names Java behaviour that does
//!   not depend on the version, such as the runtime's (`new byte[len]` throwing on a negative
//!   length).

/// A Minecraft version the client can play.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum GameVersion {
    /// Minecraft 1.7.10, protocol 5.
    V1_7_10,
    /// Minecraft 1.8.9, protocol 47: the client's own version and the default.
    #[default]
    V1_8_9,
}

impl GameVersion {
    /// Every version, oldest first (the order the title screen's slider shows them in).
    pub const ALL: [GameVersion; 2] = [GameVersion::V1_7_10, GameVersion::V1_8_9];

    /// The protocol number sent in the handshake: 5 for 1.7.10, 47 for 1.8.9.
    #[must_use]
    pub const fn protocol(self) -> i32 {
        match self {
            GameVersion::V1_7_10 => 5,
            GameVersion::V1_8_9 => 47,
        }
    }

    /// The version as players write it: `"1.7.10"` or `"1.8.9"`.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            GameVersion::V1_7_10 => "1.7.10",
            GameVersion::V1_8_9 => "1.8.9",
        }
    }

    /// The title screen's version line, the text vanilla draws in its corner.
    #[must_use]
    pub const fn title_line(self) -> &'static str {
        match self {
            GameVersion::V1_7_10 => "Minecraft 1.7.10",
            GameVersion::V1_8_9 => "Minecraft 1.8.9",
        }
    }

    /// The version named `name` (as [`GameVersion::name`] writes it), if there is one.
    #[must_use]
    pub fn from_name(name: &str) -> Option<GameVersion> {
        GameVersion::ALL
            .into_iter()
            .find(|version| version.name() == name)
    }

    /// The version that speaks protocol number `protocol`, if there is one.
    #[must_use]
    pub const fn from_protocol(protocol: i32) -> Option<GameVersion> {
        match protocol {
            5 => Some(GameVersion::V1_7_10),
            47 => Some(GameVersion::V1_8_9),
            _ => None,
        }
    }

    /// Whether this is 1.7.10, the version whose behavior differs from the client's own.
    #[must_use]
    pub const fn is_1_7_10(self) -> bool {
        matches!(self, GameVersion::V1_7_10)
    }
}

impl std::fmt::Display for GameVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_1_8_9() {
        assert_eq!(GameVersion::default(), GameVersion::V1_8_9);
        assert!(!GameVersion::default().is_1_7_10());
    }

    #[test]
    fn protocol_numbers_match_the_handshakes_of_both_versions() {
        assert_eq!(GameVersion::V1_8_9.protocol(), 47);
        assert_eq!(GameVersion::V1_7_10.protocol(), 5);
    }

    #[test]
    fn names_and_protocols_round_trip() {
        for version in GameVersion::ALL {
            assert_eq!(GameVersion::from_name(version.name()), Some(version));
            assert_eq!(
                GameVersion::from_protocol(version.protocol()),
                Some(version)
            );
            assert_eq!(version.to_string(), version.name());
            assert!(version.title_line().ends_with(version.name()));
        }
        assert_eq!(GameVersion::from_name("1.8"), None);
        assert_eq!(GameVersion::from_name(""), None);
        assert_eq!(GameVersion::from_protocol(4), None);
        assert_eq!(GameVersion::from_protocol(48), None);
    }

    #[test]
    fn all_is_oldest_first_and_sorted() {
        assert_eq!(
            GameVersion::ALL,
            [GameVersion::V1_7_10, GameVersion::V1_8_9]
        );
        assert!(GameVersion::ALL.windows(2).all(|pair| pair[0] < pair[1]));
    }
}
