//! The crate error type.
//!
//! Every crate owns one `thiserror` enum. Resource
//! packs are attacker-controlled input (a server may hand the client a `.zip` URL), so every
//! failure mode a malicious pack can reach has its own variant instead of a panic.

use std::path::Path;

use thiserror::Error;

use crate::ResourceLocation;

/// Errors returned by `mc-assets`.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum AssetError {
    /// A code path that S04/S09 have not implemented yet.
    #[error("mc-assets: {0} is not implemented yet")]
    Unimplemented(&'static str),

    /// No pack in the stack provides this resource.
    ///
    /// Java's `JsonException.func_151379_a` normalizes a missing resource's underlying
    /// `FileNotFoundException` to this exact message.  Keep the location in the typed payload
    /// for callers, but make the user-facing text match vanilla.
    #[error("File not found")]
    NotFound(ResourceLocation),

    /// A string could not be parsed as a [`ResourceLocation`].
    #[error("invalid resource location {input:?}: {reason}")]
    InvalidLocation {
        /// The string that was rejected.
        input: String,
        /// Why it was rejected.
        reason: &'static str,
    },

    /// Filesystem failure while reading a pack or an object blob.
    #[error("i/o error on {path}")]
    Io {
        /// The path being read when the failure happened.
        path: String,
        /// The underlying error.
        #[source]
        source: std::io::Error,
    },

    /// A JSON document (asset index, `pack.mcmeta`, a model file) was malformed.
    ///
    /// `what` is kept as the rendered Java `JsonException.Entry` rather than only as a bare
    /// filename. That lets callers attach the reverse-built key/index breadcrumb while the
    /// `source` remains the original `serde_json::Error` for typed cause-chain inspection.
    #[error("Invalid {what}: {message}")]
    Json {
        /// The Java-shaped file/path entry (`file key->index`, or `(Unknown file)`).
        what: String,
        /// The user-facing parser message, kept separate so internal context markers never leak.
        message: String,
        /// The underlying error.
        #[source]
        source: serde_json::Error,
    },

    /// The zip container itself is damaged or unreadable.
    #[error("zip pack {path}")]
    Zip {
        /// Path of the offending archive.
        path: String,
        /// The underlying error.
        #[source]
        source: zip::result::ZipError,
    },

    /// A zip entry name that would escape the archive root (path traversal), or that exceeds
    /// the configured entry budget. Untrusted packs are rejected rather than sanitised.
    #[error("unsafe zip pack {path}: entry {entry:?} rejected ({reason})")]
    UnsafeZipEntry {
        /// Path of the offending archive.
        path: String,
        /// The rejected entry name.
        entry: String,
        /// Why it was rejected.
        reason: &'static str,
    },

    /// A resource decompressed to more than the configured cap (zip-bomb guard).
    #[error("resource {what} exceeds the {limit} byte limit")]
    TooLarge {
        /// What was being read.
        what: String,
        /// The cap that was exceeded, in bytes.
        limit: u64,
    },

    /// `pack.mcmeta` declares a `pack_format` this client cannot read.
    #[error("unsupported pack_format {found} (this client reads {expected})")]
    PackFormat {
        /// The value found in `pack.mcmeta`.
        found: i64,
        /// The value this client supports (1 for 1.8.9).
        expected: u32,
    },

    /// `pack.mcmeta` parsed as JSON but is not shaped like pack metadata.
    #[error("malformed pack.mcmeta: {0}")]
    BadPackMeta(&'static str),

    /// The asset index parsed as JSON but is not shaped like an asset index.
    #[error("malformed asset index: {0}")]
    BadAssetIndex(String),

    // ---- S09: block models, blockstates, atlas ----
    /// A model's `parent` chain didn't resolve: a loop, excessive depth, or a parent file
    /// failed to load.
    #[error("model {location} failed to resolve: {reason}")]
    ModelResolve {
        /// The model that was being resolved.
        location: ResourceLocation,
        /// Why resolution failed.
        reason: String,
    },

    /// A texture PNG failed to decode.
    #[error("failed to decode {location} as PNG: {message}")]
    Image {
        /// The texture location being decoded.
        location: String,
        /// The underlying decoder's message.
        message: String,
    },

    /// A texture's optional `.mcmeta` sidecar was not shaped like Java's texture metadata.
    #[error("malformed texture metadata in {location}: {reason}")]
    BadTextureMeta {
        /// The texture whose sidecar was being read.
        location: String,
        /// The field or value that was rejected.
        reason: String,
    },

    /// A blockstate JSON has no entry for the variant key a state needed.
    #[error("blockstate {file} has no variant {key:?}")]
    MissingVariant {
        /// The blockstate file, e.g. `minecraft:blockstates/oak_stairs.json`.
        file: String,
        /// The variant key that was looked up.
        key: String,
    },
}

impl AssetError {
    /// Wraps an [`std::io::Error`] with the path that produced it.
    pub(crate) fn io(path: impl AsRef<Path>, source: std::io::Error) -> Self {
        Self::Io {
            path: path.as_ref().display().to_string(),
            source,
        }
    }

    /// Wraps a [`serde_json::Error`] with a description of the document.
    pub(crate) fn json(what: impl Into<String>, source: serde_json::Error) -> Self {
        let file = what.into();
        let message = source.to_string();
        Self::Json {
            what: json_entry(&file, &[]),
            message,
            source,
        }
    }

    /// Wraps a JSON error with a Java-shaped file and nested key/index breadcrumb.
    ///
    /// The path is supplied in the order in which it should be displayed. Java builds this
    /// order while unwinding by inserting each newly discovered segment at the front;
    /// json_context_prepend below exposes that same operation for recursive loaders.
    pub(crate) fn json_context(
        file: impl Into<String>,
        path: &[String],
        source: serde_json::Error,
    ) -> Self {
        let file = file.into();
        let message = source.to_string();
        Self::Json {
            what: json_entry(&file, path),
            message,
            source,
        }
    }

    /// Wraps a JSON error after prepending segments the way Java's JsonException does.
    ///
    /// The iterator is ordered from the innermost failing member outward. Each segment is
    /// inserted at the front, yielding the outer-to-inner rendering used by getMessage().
    pub(crate) fn json_context_prepend(
        file: impl Into<String>,
        inner_to_outer: impl IntoIterator<Item = String>,
        source: serde_json::Error,
    ) -> Self {
        let mut path = Vec::new();
        for segment in inner_to_outer {
            path.insert(0, segment);
        }
        Self::json_context(file, &path, source)
    }

    /// Reverse-prepend variant that keeps a cleaned display message and the original source.
    pub(crate) fn json_context_prepend_message(
        file: impl Into<String>,
        inner_to_outer: impl IntoIterator<Item = String>,
        message: impl Into<String>,
        source: serde_json::Error,
    ) -> Self {
        let mut error = Self::json_context_prepend(file, inner_to_outer, source);
        if let Self::Json {
            message: display, ..
        } = &mut error
        {
            *display = message.into();
        }
        error
    }
}

/// Renders a Java JsonException entry without requiring a second public error type.
fn json_entry(file: &str, path: &[String]) -> String {
    let file = if file.trim().is_empty() {
        "(Unknown file)"
    } else {
        file
    };
    if path.is_empty() {
        file.to_owned()
    } else {
        format!("{file} {}", path.join("->"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;

    #[test]
    fn json_context_matches_java_entry_order_and_keeps_the_cause() {
        let source = serde_json::from_str::<serde_json::Value>("{").expect_err("malformed JSON");
        let error = AssetError::json_context_prepend(
            "models/block/bad.json",
            ["texture".to_owned(), "elements[0]".to_owned()],
            source,
        );

        assert!(
            error
                .to_string()
                .starts_with("Invalid models/block/bad.json elements[0]->texture: "),
            "Java prepends each newly discovered segment: {error}"
        );
        assert!(
            error.source().is_some(),
            "the serde cause must remain inspectable"
        );
    }

    #[test]
    fn json_context_uses_unknown_file_when_no_filename_exists() {
        let source = serde_json::from_str::<serde_json::Value>("{").expect_err("malformed JSON");
        let error = AssetError::json("", source);
        assert!(
            error.to_string().starts_with("Invalid (Unknown file): "),
            "missing filenames use Java's fallback entry: {error}"
        );
    }
}
