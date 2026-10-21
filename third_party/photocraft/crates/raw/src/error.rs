//! Error type.

use std::fmt;

/// Why a raw file could not be decoded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RawError {
    /// Not a camera raw file this crate recognizes.
    NotRaw,
    /// A raw file (or a variant of one) that is recognized but not decoded yet.
    Unsupported(String),
    /// The file is damaged or inconsistent.
    Malformed(String),
    /// Declared sizes exceed the [`crate::Limits`].
    LimitExceeded(String),
}

impl RawError {
    pub(crate) fn malformed(msg: impl Into<String>) -> Self {
        RawError::Malformed(msg.into())
    }
    pub(crate) fn unsupported(msg: impl Into<String>) -> Self {
        RawError::Unsupported(msg.into())
    }
}

impl fmt::Display for RawError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RawError::NotRaw => write!(f, "not a camera raw file"),
            RawError::Unsupported(m) => write!(f, "unsupported camera raw: {m}"),
            RawError::Malformed(m) => write!(f, "malformed camera raw: {m}"),
            RawError::LimitExceeded(m) => write!(f, "camera raw exceeds limits: {m}"),
        }
    }
}

impl std::error::Error for RawError {}

pub(crate) type Result<T> = std::result::Result<T, RawError>;
