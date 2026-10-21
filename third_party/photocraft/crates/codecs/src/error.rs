use crate::format::Format;

/// Errors produced by this crate. Decoders never panic on malformed input;
/// they return [`CodecError::Malformed`] instead.
#[derive(Debug, thiserror::Error)]
pub enum CodecError {
    #[error("unrecognized image format")]
    UnknownFormat,
    #[error("{format:?}: {reason}")]
    Unsupported { format: Format, reason: String },
    #[error("{format:?}: malformed data: {message}")]
    Malformed { format: Format, message: String },
    #[error("limit exceeded: {0}")]
    LimitExceeded(String),
    #[error("{format:?}: encode failed: {message}")]
    Encode { format: Format, message: String },
    #[error("invalid image: {0}")]
    InvalidImage(String),
}

impl CodecError {
    pub(crate) fn malformed(format: Format, e: impl std::fmt::Display) -> Self {
        CodecError::Malformed { format, message: e.to_string() }
    }
    pub(crate) fn encode(format: Format, e: impl std::fmt::Display) -> Self {
        CodecError::Encode { format, message: e.to_string() }
    }
    pub(crate) fn unsupported(format: Format, reason: impl Into<String>) -> Self {
        CodecError::Unsupported { format, reason: reason.into() }
    }
}
