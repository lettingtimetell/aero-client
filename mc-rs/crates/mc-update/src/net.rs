//! HTTP for the updater: HTTPS to GitHub's own hosts only, every redirect hop checked, every
//! body capped, certificates always verified (reqwest's rustls with its bundled roots; nothing
//! here can turn the checks off).

use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

use crate::error::UpdateError;
use crate::verify::hex;

/// The hosts the updater may talk to: the API, the release pages, and the hosts GitHub's release
/// downloads redirect to. Exact names, no wildcards.
pub const GITHUB_HOSTS: &[&str] = &[
    "api.github.com",
    "github.com",
    "objects.githubusercontent.com",
    "release-assets.githubusercontent.com",
    "github-releases.githubusercontent.com",
];

/// The API's base URL.
pub const API_BASE: &str = "https://api.github.com";

/// Redirect hops followed before giving up (a release download takes one).
pub const MAX_REDIRECTS: usize = 5;

/// The largest release JSON read (a release with a long changelog is tens of kilobytes).
pub const MAX_API_BYTES: u64 = 1 << 20;

/// The largest `.sha256` or `.sig` asset read.
pub const MAX_SIDECAR_BYTES: u64 = 4096;

/// The User-Agent: the client's name and version, nothing about the player or the PC.
pub const USER_AGENT: &str = concat!("Aero-Client-Updater/", env!("CARGO_PKG_VERSION"));

const CONNECT_TIMEOUT: Duration = Duration::from_secs(15);
/// No byte for this long ends a request.
const READ_TIMEOUT: Duration = Duration::from_secs(60);
const API_TIMEOUT: Duration = Duration::from_secs(30);
/// A whole download (about 20 MB; this allows a very slow connection).
const DOWNLOAD_TIMEOUT: Duration = Duration::from_secs(30 * 60);

/// Which URLs the updater may fetch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HttpPolicy {
    api_base: String,
    hosts: Vec<String>,
    /// Tests only: plain HTTP to the loopback mock server.
    loopback_http: bool,
}

impl HttpPolicy {
    /// The shipped policy: `https://` on [`GITHUB_HOSTS`], default port.
    #[must_use]
    pub fn github() -> HttpPolicy {
        HttpPolicy {
            api_base: API_BASE.to_owned(),
            hosts: GITHUB_HOSTS.iter().map(|host| (*host).to_owned()).collect(),
            loopback_http: false,
        }
    }

    /// Tests only: plain HTTP to `127.0.0.1` (a local mock server, any port) and nothing else.
    /// Not compiled into the shipped client.
    #[cfg(test)]
    pub(crate) fn loopback(api_base: &str) -> HttpPolicy {
        HttpPolicy {
            api_base: api_base.trim_end_matches('/').to_owned(),
            hosts: vec!["127.0.0.1".to_owned()],
            loopback_http: true,
        }
    }

    /// The API's base URL, without a trailing slash.
    #[must_use]
    pub fn api_base(&self) -> &str {
        &self.api_base
    }

    /// `url` parsed, if the policy allows it.
    pub fn check_url(&self, url: &str) -> Result<reqwest::Url, UpdateError> {
        let parsed =
            reqwest::Url::parse(url).map_err(|_| UpdateError::RefusedUrl(short_url_text(url)))?;
        self.check_parsed(&parsed)?;
        Ok(parsed)
    }

    /// Whether the policy allows `url`: no credentials in it, an allowed host, and `https` on the
    /// default port (the test policy: `http` to the loopback host).
    pub fn check_parsed(&self, url: &reqwest::Url) -> Result<(), UpdateError> {
        let refuse = || UpdateError::RefusedUrl(short_url(url));
        if !url.username().is_empty() || url.password().is_some() {
            return Err(refuse());
        }
        let host = url.host_str().ok_or_else(refuse)?;
        if !self.hosts.iter().any(|allowed| allowed == host) {
            return Err(refuse());
        }
        match url.scheme() {
            "https" if url.port().is_none() => Ok(()),
            "http" if self.loopback_http => Ok(()),
            _ => Err(refuse()),
        }
    }
}

/// A URL for a log line: scheme, host and path, never the query (GitHub's download redirects
/// carry signed query strings).
fn short_url(url: &reqwest::Url) -> String {
    let mut text = format!(
        "{}://{}{}",
        url.scheme(),
        url.host_str().unwrap_or(""),
        url.path()
    );
    cut_at_char(&mut text, 160);
    text
}

fn short_url_text(url: &str) -> String {
    let mut text: String = url.split(['?', '#']).next().unwrap_or("").to_owned();
    cut_at_char(&mut text, 160);
    text
}

/// Cuts `text` to at most `max` bytes at a character boundary.
/// `String::truncate` panics inside a character, and a URL that failed to parse is the release
/// data's raw text (a non-ASCII one panicked the updater's thread).
fn cut_at_char(text: &mut String, max: usize) {
    if text.len() > max {
        let mut cut = max;
        while !text.is_char_boundary(cut) {
            cut -= 1;
        }
        text.truncate(cut);
    }
}

fn network(error: reqwest::Error) -> UpdateError {
    UpdateError::Network(error.without_url().to_string())
}

/// The HTTP client for one policy: rustls with its bundled roots, [`USER_AGENT`], timeouts, and
/// a redirect policy that checks every hop against `policy` (plus `https_only` in the shipped
/// policy, so a redirect to `http://` fails even before the hop check).
pub fn client(policy: &HttpPolicy) -> Result<reqwest::Client, UpdateError> {
    let redirects = policy.clone();
    reqwest::Client::builder()
        .user_agent(USER_AGENT)
        .connect_timeout(CONNECT_TIMEOUT)
        .read_timeout(READ_TIMEOUT)
        .https_only(!policy.loopback_http)
        .redirect(reqwest::redirect::Policy::custom(move |attempt| {
            if attempt.previous().len() > MAX_REDIRECTS {
                attempt.error("too many redirects")
            } else if redirects.check_parsed(attempt.url()).is_ok() {
                attempt.follow()
            } else {
                attempt.error("a redirect left the update hosts")
            }
        }))
        .build()
        .map_err(network)
}

/// The latest release's JSON: `Ok(None)` when the repository has no release (or is not public:
/// GitHub answers 404 for both).
pub async fn fetch_latest_release(
    client: &reqwest::Client,
    policy: &HttpPolicy,
    repo: &str,
) -> Result<Option<Vec<u8>>, UpdateError> {
    let url = policy.check_url(&format!(
        "{}/repos/{repo}/releases/latest",
        policy.api_base()
    ))?;
    let response = client
        .get(url)
        .header(reqwest::header::ACCEPT, "application/vnd.github+json")
        .header("X-GitHub-Api-Version", "2022-11-28")
        .timeout(API_TIMEOUT)
        .send()
        .await
        .map_err(network)?;
    match response.status().as_u16() {
        200 => {}
        404 => return Ok(None),
        403 | 429 => return Err(UpdateError::RateLimited),
        status => return Err(UpdateError::Status(status)),
    }
    read_capped(response, MAX_API_BYTES).await.map(Some)
}

/// A small asset (`.sha256`, `.sig`): at most [`MAX_SIDECAR_BYTES`].
pub async fn fetch_small(
    client: &reqwest::Client,
    policy: &HttpPolicy,
    url: &str,
) -> Result<Vec<u8>, UpdateError> {
    let url = policy.check_url(url)?;
    let response = client
        .get(url)
        .timeout(API_TIMEOUT)
        .send()
        .await
        .map_err(network)?;
    match response.status().as_u16() {
        200 => read_capped(response, MAX_SIDECAR_BYTES).await,
        status => Err(UpdateError::Status(status)),
    }
}

/// A body of at most `cap` bytes, whatever `Content-Length` claims.
async fn read_capped(mut response: reqwest::Response, cap: u64) -> Result<Vec<u8>, UpdateError> {
    if response.content_length().is_some_and(|length| length > cap) {
        return Err(UpdateError::Size(format!("a response over {cap} bytes")));
    }
    let mut body = Vec::new();
    while let Some(chunk) = response.chunk().await.map_err(network)? {
        if body.len() as u64 + chunk.len() as u64 > cap {
            return Err(UpdateError::Size(format!("a response over {cap} bytes")));
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body)
}

/// A download's progress, read by the title screen (two relaxed loads per frame while shown).
#[derive(Debug, Default)]
pub struct Progress {
    /// Bytes received.
    pub done: AtomicU64,
    /// Bytes expected.
    pub total: AtomicU64,
}

/// Download `url` into `dest` (created or truncated), exactly `size` bytes: a `Content-Length`
/// that differs, a longer body or a shorter one is an error. Returns the body's SHA-256 (lower
/// case hex). `cancel` is checked between chunks.
pub async fn download(
    client: &reqwest::Client,
    policy: &HttpPolicy,
    url: &str,
    size: u64,
    dest: &Path,
    progress: &Progress,
    cancel: &AtomicBool,
) -> Result<String, UpdateError> {
    let url = policy.check_url(url)?;
    progress.total.store(size, Ordering::Relaxed);
    progress.done.store(0, Ordering::Relaxed);
    let mut response = client
        .get(url)
        .timeout(DOWNLOAD_TIMEOUT)
        .send()
        .await
        .map_err(network)?;
    let status = response.status().as_u16();
    if status != 200 {
        return Err(UpdateError::Status(status));
    }
    if response
        .content_length()
        .is_some_and(|length| length != size)
    {
        return Err(UpdateError::Size(
            "the download's length is not the release's size".to_owned(),
        ));
    }
    let file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(dest)
        .map_err(|error| UpdateError::io("creating the download file", error))?;
    let mut out = BufWriter::with_capacity(1 << 16, file);
    let mut hasher = ring::digest::Context::new(&ring::digest::SHA256);
    let mut done = 0_u64;
    while let Some(chunk) = response.chunk().await.map_err(network)? {
        if cancel.load(Ordering::Relaxed) {
            return Err(UpdateError::Cancelled);
        }
        done += chunk.len() as u64;
        if done > size {
            return Err(UpdateError::Size(
                "the download is longer than the release's size".to_owned(),
            ));
        }
        hasher.update(&chunk);
        out.write_all(&chunk)
            .map_err(|error| UpdateError::io("writing the download", error))?;
        progress.done.store(done, Ordering::Relaxed);
    }
    if done != size {
        return Err(UpdateError::Size(
            "the download is shorter than the release's size".to_owned(),
        ));
    }
    let file = out
        .into_inner()
        .map_err(|error| UpdateError::io("writing the download", error.into_error()))?;
    file.sync_all()
        .map_err(|error| UpdateError::io("writing the download", error))?;
    Ok(hex(hasher.finish().as_ref()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_shipped_policy_allows_https_on_github_hosts_only() {
        let policy = HttpPolicy::github();
        for ok in [
            "https://api.github.com/repos/o/r/releases/latest",
            "https://github.com/o/r/releases/download/v0.2.0/a.zip",
            "https://objects.githubusercontent.com/x",
            "https://release-assets.githubusercontent.com/x?sig=abc",
            "https://GitHub.com/o/r",
        ] {
            assert!(policy.check_url(ok).is_ok(), "{ok}");
        }
        for refused in [
            "http://github.com/o/r",
            "http://api.github.com/x",
            "https://github.com:8443/o/r",
            "https://evil.example/x",
            "https://github.com.evil.example/x",
            "https://evilgithub.com/x",
            "https://user:pass@github.com/x",
            "https://user@github.com/x",
            "https://raw.githubusercontent.com/x",
            "https://gist.github.com/x",
            "https://127.0.0.1/x",
            "https://[::1]/x",
            "file:///C:/x",
            "ftp://github.com/x",
            "github.com/o/r",
            "",
        ] {
            assert!(policy.check_url(refused).is_err(), "{refused}");
        }
    }

    #[test]
    fn the_test_policy_allows_only_plain_loopback() {
        let policy = HttpPolicy::loopback("http://127.0.0.1:4000/");
        assert_eq!(policy.api_base(), "http://127.0.0.1:4000");
        assert!(policy.check_url("http://127.0.0.1:4000/x").is_ok());
        assert!(policy.check_url("http://localhost:4000/x").is_err());
        assert!(policy.check_url("http://github.com/x").is_err());
        assert!(policy.check_url("https://github.com/x").is_err());
    }

    #[test]
    fn log_text_never_keeps_the_query() {
        let url =
            reqwest::Url::parse("https://release-assets.githubusercontent.com/a/b?sig=SECRET")
                .unwrap();
        assert_eq!(
            short_url(&url),
            "https://release-assets.githubusercontent.com/a/b"
        );
        assert_eq!(
            short_url_text("https://x.example/p?sig=SECRET#f"),
            "https://x.example/p"
        );
    }

    #[test]
    fn the_user_agent_names_the_client_only() {
        assert_eq!(
            USER_AGENT,
            format!("Aero-Client-Updater/{}", crate::CURRENT_VERSION)
        );
    }
}
