//! Errors.

/// Why an encode failed.
#[derive(Debug, thiserror::Error)]
pub enum VideoError {
    /// Media Foundation (or the requested codec) is not available on this PC.
    #[error("video export is not available: {0}")]
    Unavailable(String),
    /// The configuration cannot work.
    #[error("invalid export settings: {0}")]
    InvalidConfig(String),
    /// A Media Foundation call failed.
    #[error("Media Foundation {context} failed: {message} (0x{hresult:08X})")]
    MediaFoundation {
        /// What was being done.
        context: &'static str,
        /// The HRESULT.
        hresult: u32,
        /// The system's text for it.
        message: String,
    },
    /// A frame of the wrong size was submitted.
    #[error("frame is {got} bytes, expected {expected}")]
    FrameSize {
        /// Bytes handed in.
        got: usize,
        /// Bytes an NV12 frame of the configured size has.
        expected: usize,
    },
    /// File system trouble.
    #[error("file error: {0}")]
    Io(#[from] std::io::Error),
    /// The encode was cancelled.
    #[error("cancelled")]
    Cancelled,
    /// The worker thread stopped unexpectedly.
    #[error("the encoder thread stopped: {0}")]
    WorkerGone(String),
}

#[cfg(windows)]
impl VideoError {
    /// Wrap a `windows` error with what was being attempted.
    pub(crate) fn mf(context: &'static str, error: &windows::core::Error) -> Self {
        Self::MediaFoundation {
            context,
            hresult: error.code().0 as u32,
            message: error.message().trim().to_owned(),
        }
    }

    /// The HRESULT, for a Media Foundation failure.
    #[must_use]
    pub fn hresult(&self) -> Option<u32> {
        match self {
            Self::MediaFoundation { hresult, .. } => Some(*hresult),
            _ => None,
        }
    }
}
