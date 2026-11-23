//! Release tags and version order.

use std::cmp::Ordering;

pub use semver::Version;

/// The longest tag accepted.
const MAX_TAG_LEN: usize = 64;

/// The version in a release tag, `v0.2.0` or `0.2.0`. `None` for anything else, and for a
/// prerelease version (`v0.2.0-beta.1`): only stable releases are offered.
#[must_use]
pub fn parse_release_tag(tag: &str) -> Option<Version> {
    if tag.is_empty() || tag.len() > MAX_TAG_LEN {
        return None;
    }
    let bare = tag
        .strip_prefix('v')
        .or_else(|| tag.strip_prefix('V'))
        .unwrap_or(tag);
    let version = Version::parse(bare).ok()?;
    version.pre.is_empty().then_some(version)
}

/// This build's version ([`crate::CURRENT_VERSION`]).
#[must_use]
pub fn current() -> Version {
    // Cargo only accepts a semver package version, so this cannot fail.
    Version::parse(crate::CURRENT_VERSION).unwrap_or_else(|_| Version::new(0, 0, 0))
}

/// Whether `candidate` is a newer release than `current`: semver precedence, build metadata
/// ignored.
#[must_use]
pub fn is_newer(candidate: &Version, current: &Version) -> bool {
    candidate.cmp_precedence(current) == Ordering::Greater
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tags_parse_with_or_without_the_v() {
        assert_eq!(parse_release_tag("v0.2.0"), Some(Version::new(0, 2, 0)));
        assert_eq!(parse_release_tag("0.2.0"), Some(Version::new(0, 2, 0)));
        assert_eq!(parse_release_tag("V1.10.3"), Some(Version::new(1, 10, 3)));
        assert_eq!(
            parse_release_tag("v0.2.0+build.7").map(|v| (v.major, v.minor, v.patch)),
            Some((0, 2, 0))
        );
    }

    #[test]
    fn malformed_and_prerelease_tags_are_refused() {
        for tag in [
            "",
            "v",
            "latest",
            "v0.2",
            "v0.2.0.1",
            "vv0.2.0",
            " v0.2.0",
            "v0.2.0 ",
            "v01.2.0",
            "v0.2.0-beta.1",
            "v0.2.0-rc1",
            "release-0.2.0",
            &"9".repeat(80),
        ] {
            assert_eq!(parse_release_tag(tag), None, "{tag:?}");
        }
    }

    #[test]
    fn newer_means_semver_precedence() {
        let current = Version::new(0, 1, 0);
        let newer = |tag: &str| is_newer(&parse_release_tag(tag).expect(tag), &current);
        assert!(newer("v0.2.0"));
        assert!(newer("v0.1.1"));
        assert!(newer("v1.0.0"));
        assert!(newer("v0.10.0"), "numeric, not lexical");
        assert!(!newer("v0.1.0"), "the same version is not an update");
        assert!(!newer("v0.0.9"), "never a downgrade");
        assert!(!newer("v0.1.0+other"), "build metadata alone is not newer");
        let ten = Version::new(0, 10, 0);
        assert!(!is_newer(&Version::new(0, 9, 0), &ten));
    }

    #[test]
    fn the_current_version_is_the_workspace_version() {
        assert_eq!(current().to_string(), crate::CURRENT_VERSION);
    }
}
