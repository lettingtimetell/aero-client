//! The release data GitHub returns, and choosing what to offer from it.
//!
//! `GET /repos/<owner>/<repo>/releases/latest` returns the newest release that is neither a
//! draft nor a prerelease; [`choose_release`] checks both flags again anyway, then the tag, the
//! Windows portable zip ([`asset_name`]), its size and URL, and how it can be verified.

use serde::{Deserialize, Serialize};

use crate::error::UpdateError;
use crate::net::{HttpPolicy, MAX_SIDECAR_BYTES};
use crate::version::{self, Version};

/// The largest release zip accepted (today's is about 20 MB).
pub const MAX_ZIP_BYTES: u64 = 256 * 1024 * 1024;

/// The Windows portable zip's asset name for `version`: `Aero-Client-0.2.0-windows-x64.zip`.
#[must_use]
pub fn asset_name(version: &Version) -> String {
    format!("Aero-Client-{version}-windows-x64.zip")
}

/// The checksum asset's name: `<asset>.sha256`.
#[must_use]
pub fn sha256_asset_name(asset: &str) -> String {
    format!("{asset}.sha256")
}

/// The signature asset's name: `<asset>.sig`.
#[must_use]
pub fn signature_asset_name(asset: &str) -> String {
    format!("{asset}.sig")
}

/// Whether `repo` is a plausible `owner/name` (GitHub's character sets and lengths), so it can
/// go into a URL path as it is.
#[must_use]
pub fn valid_repo(repo: &str) -> bool {
    let Some((owner, name)) = repo.split_once('/') else {
        return false;
    };
    let owner_ok = (1..=39).contains(&owner.len())
        && owner
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-');
    let name_ok = (1..=100).contains(&name.len())
        && name != "."
        && name != ".."
        && name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'));
    owner_ok && name_ok
}

/// A release as the API returns it (only the fields used).
#[derive(Debug, Clone, Deserialize)]
pub struct ApiRelease {
    /// The tag, `v0.2.0`.
    pub tag_name: String,
    /// A draft (never returned by `/releases/latest`; checked anyway).
    #[serde(default)]
    pub draft: bool,
    /// A prerelease (never returned by `/releases/latest`; checked anyway).
    #[serde(default)]
    pub prerelease: bool,
    /// The release page.
    #[serde(default)]
    pub html_url: Option<String>,
    /// The files attached.
    #[serde(default)]
    pub assets: Vec<ApiAsset>,
}

/// One attached file.
#[derive(Debug, Clone, Deserialize)]
pub struct ApiAsset {
    /// File name.
    pub name: String,
    /// Bytes.
    pub size: u64,
    /// Where browsers download it (redirects to GitHub's download host).
    pub browser_download_url: String,
    /// `sha256:<hex>`, which GitHub computes for uploaded assets.
    #[serde(default)]
    pub digest: Option<String>,
    /// `uploaded` once complete.
    #[serde(default)]
    pub state: Option<String>,
}

/// A downloadable file of a release.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AssetRef {
    /// File name.
    pub name: String,
    /// Download URL (checked against the policy when chosen and again when fetched).
    pub url: String,
    /// Bytes.
    pub size: u64,
}

/// A release worth offering, as the updater keeps it (also in `state.json`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReleaseInfo {
    /// The version, `0.2.0`.
    pub version: String,
    /// The tag, `v0.2.0`.
    pub tag: String,
    /// The release page, if it is on `https://github.com/`.
    pub notes_url: Option<String>,
    /// The Windows portable zip.
    pub zip: AssetRef,
    /// Its SHA-256 from the API's `digest`, when GitHub gave one.
    pub sha256: Option<String>,
    /// `<asset>.sha256`, when attached.
    pub sha256_asset: Option<AssetRef>,
    /// `<asset>.sig`, when attached.
    pub signature_asset: Option<AssetRef>,
}

/// Parse the API's JSON.
pub fn parse_release(json: &[u8]) -> Result<ApiRelease, UpdateError> {
    serde_json::from_slice(json)
        .map_err(|_| UpdateError::BadRelease("the release data is not a release".to_owned()))
}

/// What to offer from `release`: `Ok(None)` when it is not an update for `current` (a draft, a
/// prerelease, an unusable tag, not newer, or no Windows portable zip attached); an error when it
/// is newer but cannot be installed safely (an oversized or off-policy asset, a malformed digest,
/// no way to check its hash, or no signature while `signature_required`).
pub fn choose_release(
    release: &ApiRelease,
    current: &Version,
    policy: &HttpPolicy,
    signature_required: bool,
) -> Result<Option<ReleaseInfo>, UpdateError> {
    if release.draft || release.prerelease {
        return Ok(None);
    }
    let Some(version) = version::parse_release_tag(&release.tag_name) else {
        return Ok(None);
    };
    if !version::is_newer(&version, current) {
        return Ok(None);
    }
    let name = asset_name(&version);
    let Some(zip) = release.assets.iter().find(|asset| asset.name == name) else {
        return Ok(None);
    };
    if zip
        .state
        .as_deref()
        .is_some_and(|state| state != "uploaded")
    {
        return Ok(None);
    }
    if zip.size == 0 || zip.size > MAX_ZIP_BYTES {
        return Err(UpdateError::Size(format!(
            "the release zip is {} bytes",
            zip.size
        )));
    }
    policy.check_url(&zip.browser_download_url)?;
    let sha256 = zip
        .digest
        .as_deref()
        .map(crate::verify::parse_api_digest)
        .transpose()?;
    let sidecar = |sidecar_name: String| -> Result<Option<AssetRef>, UpdateError> {
        let Some(asset) = release
            .assets
            .iter()
            .find(|asset| asset.name == sidecar_name)
        else {
            return Ok(None);
        };
        if asset.size == 0 || asset.size > MAX_SIDECAR_BYTES {
            return Err(UpdateError::Size(format!(
                "{sidecar_name} is {} bytes",
                asset.size
            )));
        }
        policy.check_url(&asset.browser_download_url)?;
        Ok(Some(AssetRef {
            name: asset.name.clone(),
            url: asset.browser_download_url.clone(),
            size: asset.size,
        }))
    };
    let sha256_asset = sidecar(sha256_asset_name(&name))?;
    let signature_asset = sidecar(signature_asset_name(&name))?;
    if sha256.is_none() && sha256_asset.is_none() {
        return Err(UpdateError::BadRelease(
            "the release zip has no digest and no .sha256 asset".to_owned(),
        ));
    }
    if signature_required && signature_asset.is_none() {
        return Err(UpdateError::Signature(
            "this build needs a signed release and the .sig asset is missing".to_owned(),
        ));
    }
    let notes_url = release
        .html_url
        .as_deref()
        .filter(|url| release_page_ok(url))
        .map(str::to_owned);
    Ok(Some(ReleaseInfo {
        version: version.to_string(),
        tag: release.tag_name.clone(),
        notes_url,
        zip: AssetRef {
            name,
            url: zip.browser_download_url.clone(),
            size: zip.size,
        },
        sha256,
        sha256_asset,
        signature_asset,
    }))
}

/// Whether `url` may be opened in the player's browser as release notes: `https://github.com/`
/// and nothing unusual in it.
#[must_use]
pub fn release_page_ok(url: &str) -> bool {
    url.starts_with("https://github.com/")
        && url.len() <= 300
        && url
            .bytes()
            .all(|byte| byte.is_ascii_graphic() && !matches!(byte, b'"' | b'<' | b'>' | b'\\'))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const BASE: &str = "https://github.com/o/r/releases/download/v0.2.0";

    fn release(extra_assets: serde_json::Value) -> serde_json::Value {
        let mut assets = vec![json!({
            "name": "Aero-Client-0.2.0-windows-x64.zip",
            "size": 20_000_000,
            "browser_download_url": format!("{BASE}/Aero-Client-0.2.0-windows-x64.zip"),
            "digest": format!("sha256:{}", "ab".repeat(32)),
            "state": "uploaded"
        })];
        if let serde_json::Value::Array(more) = extra_assets {
            assets.extend(more);
        }
        json!({
            "tag_name": "v0.2.0",
            "draft": false,
            "prerelease": false,
            "html_url": "https://github.com/o/r/releases/tag/v0.2.0",
            "assets": assets
        })
    }

    fn choose(value: &serde_json::Value, signed: bool) -> Result<Option<ReleaseInfo>, UpdateError> {
        let release = parse_release(value.to_string().as_bytes()).unwrap();
        choose_release(
            &release,
            &Version::new(0, 1, 0),
            &HttpPolicy::github(),
            signed,
        )
    }

    #[test]
    fn the_asset_names_follow_the_convention() {
        let version = Version::new(0, 2, 0);
        assert_eq!(asset_name(&version), "Aero-Client-0.2.0-windows-x64.zip");
        assert_eq!(
            sha256_asset_name(&asset_name(&version)),
            "Aero-Client-0.2.0-windows-x64.zip.sha256"
        );
        assert_eq!(
            signature_asset_name(&asset_name(&version)),
            "Aero-Client-0.2.0-windows-x64.zip.sig"
        );
    }

    #[test]
    fn a_newer_release_with_its_zip_is_offered() {
        let info = choose(&release(json!([])), false)
            .unwrap()
            .expect("offered");
        assert_eq!(info.version, "0.2.0");
        assert_eq!(info.tag, "v0.2.0");
        assert_eq!(info.zip.name, "Aero-Client-0.2.0-windows-x64.zip");
        assert_eq!(info.zip.size, 20_000_000);
        assert_eq!(info.sha256.as_deref(), Some("ab".repeat(32).as_str()));
        assert_eq!(
            info.notes_url.as_deref(),
            Some("https://github.com/o/r/releases/tag/v0.2.0")
        );
        assert!(info.signature_asset.is_none());
    }

    #[test]
    fn drafts_prereleases_old_versions_and_other_assets_are_not_offered() {
        let mut value = release(json!([]));
        value["draft"] = json!(true);
        assert_eq!(choose(&value, false).unwrap(), None, "draft");
        let mut value = release(json!([]));
        value["prerelease"] = json!(true);
        assert_eq!(choose(&value, false).unwrap(), None, "prerelease");
        for tag in ["v0.1.0", "v0.0.9", "v0.2.0-beta.1", "nightly"] {
            let mut value = release(json!([]));
            value["tag_name"] = json!(tag);
            assert_eq!(choose(&value, false).unwrap(), None, "{tag}");
        }
        // Only a zip for another version, or another platform: nothing to install.
        let mut value = release(json!([]));
        value["assets"][0]["name"] = json!("Aero-Client-0.1.9-windows-x64.zip");
        assert_eq!(choose(&value, false).unwrap(), None);
        let mut value = release(json!([]));
        value["assets"][0]["name"] = json!("Aero-Client-0.2.0-linux-x64.tar.gz");
        assert_eq!(choose(&value, false).unwrap(), None);
        let mut value = release(json!([]));
        value["assets"][0]["state"] = json!("starter");
        assert_eq!(choose(&value, false).unwrap(), None, "upload not finished");
    }

    #[test]
    fn unsafe_releases_are_refused_with_an_error() {
        // Off-policy download URLs.
        for url in [
            "http://github.com/o/r/a.zip",
            "https://evil.example/a.zip",
            "https://github.com:444/a.zip",
        ] {
            let mut value = release(json!([]));
            value["assets"][0]["browser_download_url"] = json!(url);
            assert!(choose(&value, false).is_err(), "{url}");
        }
        // Sizes: empty, over the cap.
        for size in [0_u64, MAX_ZIP_BYTES + 1] {
            let mut value = release(json!([]));
            value["assets"][0]["size"] = json!(size);
            assert!(matches!(choose(&value, false), Err(UpdateError::Size(_))));
        }
        // A malformed digest is an error, not "no digest".
        let mut value = release(json!([]));
        value["assets"][0]["digest"] = json!("md5:1234");
        assert!(choose(&value, false).is_err());
        // No digest and no .sha256: no way to check the download.
        let mut value = release(json!([]));
        value["assets"][0]["digest"] = serde_json::Value::Null;
        assert!(matches!(
            choose(&value, false),
            Err(UpdateError::BadRelease(_))
        ));
    }

    #[test]
    fn a_sha256_asset_stands_in_for_the_digest() {
        let mut value = release(json!([{
            "name": "Aero-Client-0.2.0-windows-x64.zip.sha256",
            "size": 100,
            "browser_download_url": format!("{BASE}/Aero-Client-0.2.0-windows-x64.zip.sha256")
        }]));
        value["assets"][0]["digest"] = serde_json::Value::Null;
        let info = choose(&value, false).unwrap().expect("offered");
        assert_eq!(info.sha256, None);
        assert_eq!(
            info.sha256_asset.map(|asset| asset.name),
            Some("Aero-Client-0.2.0-windows-x64.zip.sha256".to_owned())
        );
        // A huge "checksum" is refused.
        value["assets"][1]["size"] = json!(MAX_SIDECAR_BYTES + 1);
        assert!(choose(&value, false).is_err());
    }

    #[test]
    fn a_build_with_a_key_needs_the_signature_asset() {
        let unsigned = release(json!([]));
        assert!(matches!(
            choose(&unsigned, true),
            Err(UpdateError::Signature(_))
        ));
        let signed = release(json!([{
            "name": "Aero-Client-0.2.0-windows-x64.zip.sig",
            "size": 89,
            "browser_download_url": format!("{BASE}/Aero-Client-0.2.0-windows-x64.zip.sig")
        }]));
        let info = choose(&signed, true).unwrap().expect("offered");
        assert!(info.signature_asset.is_some());
    }

    #[test]
    fn release_pages_must_be_on_github() {
        assert!(release_page_ok(
            "https://github.com/o/r/releases/tag/v0.2.0"
        ));
        for bad in [
            "http://github.com/o/r",
            "https://github.com.evil.example/x",
            "https://evil.example/https://github.com/",
            "javascript:alert(1)",
            "https://github.com/o/r\" & calc",
            "https://github.com/o/r with space",
        ] {
            assert!(!release_page_ok(bad), "{bad}");
        }
        let mut value = release(json!([]));
        value["html_url"] = json!("https://evil.example/notes");
        assert_eq!(choose(&value, false).unwrap().unwrap().notes_url, None);
    }

    #[test]
    fn repository_names_are_checked_before_they_go_into_a_url() {
        assert!(valid_repo(crate::DEFAULT_REPO));
        assert!(valid_repo(crate::REPO), "the build's repository is usable");
        assert!(valid_repo("a-b/c.d_e"));
        for bad in [
            "", "owner", "/name", "owner/", "a/b/c", "a/..", "a b/c", "a/c?x", "a/c#x", "../x",
        ] {
            assert!(!valid_repo(bad), "{bad}");
        }
    }
}
