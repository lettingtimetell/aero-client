//! The crate's one error type.

/// Why a replay file could not be read or written.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ReplayError {
    /// The operating system refused a read or a write.
    #[error("replay file I/O: {0}")]
    Io(#[from] std::io::Error),
    /// The file does not start with the replay magic, or is too short to be one.
    #[error("not an Aero replay file")]
    NotAReplay,
    /// The file is from a newer client.
    #[error("replay container version {0} is newer than this client understands")]
    UnsupportedVersion(u16),
    /// The metadata JSON is damaged or of the wrong shape.
    #[error("replay metadata: {0}")]
    Meta(String),
    /// A block or the footer is damaged (a checksum, a length or a record does not add up).
    #[error("damaged replay data: {0}")]
    Corrupt(String),
    /// Holding the file's records would take far more memory than a recording of its size can
    /// (a damaged or crafted file): refused while it is read.
    #[error("replay is too large to open: {0}")]
    TooLarge(String),
}
