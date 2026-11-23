//! The updater's error type.

/// Why an update step failed. The `Display` text is for the log: it names no local folder and
/// holds no secret (the updater has none). [`UpdateError::player_message`] is for the screen.
#[derive(Debug, thiserror::Error)]
pub enum UpdateError {
    /// The request failed: no connection, TLS, a timeout.
    #[error("network error: {0}")]
    Network(String),
    /// The server answered with an unexpected HTTP status.
    #[error("HTTP status {0}")]
    Status(u16),
    /// The release server's rate limit (HTTP 403 or 429).
    #[error("rate limited by the release server")]
    RateLimited,
    /// A URL that is not HTTPS on one of the allowed hosts (or a redirect to one).
    #[error("refused URL: {0}")]
    RefusedUrl(String),
    /// More bytes than allowed, or a different length than the release data says.
    #[error("size: {0}")]
    Size(String),
    /// The release's data is unusable.
    #[error("unusable release: {0}")]
    BadRelease(String),
    /// The download's SHA-256 is not the published one.
    #[error("SHA-256 mismatch")]
    ChecksumMismatch,
    /// The signature is missing, malformed or does not verify.
    #[error("signature: {0}")]
    Signature(String),
    /// The archive is malformed or would write outside its folder.
    #[error("archive: {0}")]
    Archive(String),
    /// A file operation failed. `context` names the step relative to the install folder.
    #[error("{context}: {source}")]
    Io {
        /// What was being done.
        context: String,
        /// The OS error.
        #[source]
        source: std::io::Error,
    },
    /// This build is not a packaged portable install, or the install is incomplete.
    #[error("not a portable install: {0}")]
    NotPortable(String),
    /// The player cancelled the download.
    #[error("cancelled")]
    Cancelled,
    /// The update cannot be applied in this state.
    #[error("{0}")]
    Apply(String),
}

impl UpdateError {
    /// An I/O error with what was being done.
    pub(crate) fn io(context: impl Into<String>, source: std::io::Error) -> UpdateError {
        UpdateError::Io {
            context: context.into(),
            source,
        }
    }

    /// One short sentence for the title screen's update card.
    #[must_use]
    pub fn player_message(&self) -> &'static str {
        match self {
            UpdateError::Network(_) | UpdateError::Status(_) | UpdateError::RateLimited => {
                "Couldn't reach GitHub. Try again later."
            }
            UpdateError::Size(_) | UpdateError::ChecksumMismatch => {
                "The download was damaged. Try again."
            }
            UpdateError::Signature(_) => {
                "The download isn't signed by Aero Client, so it wasn't used."
            }
            UpdateError::RefusedUrl(_) | UpdateError::BadRelease(_) | UpdateError::Archive(_) => {
                "This release can't be installed here. Get it from the release page."
            }
            UpdateError::Io { .. } => "Couldn't save the update next to Aero Client.",
            UpdateError::NotPortable(_) => "Only the portable Aero Client updates itself.",
            UpdateError::Cancelled => "Download cancelled.",
            UpdateError::Apply(_) => "The update couldn't be applied.",
        }
    }
}
