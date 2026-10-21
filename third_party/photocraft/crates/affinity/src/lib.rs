//! Reader for native Affinity documents (`.af`, and `.afdesign`/`.afphoto`/`.afpub` from
//! Affinity 1 and 2): the archive container, the tagged object stream in `doc.dat`, and the
//! subset of the document model PhotoCraft imports.
//!
//! Layouts follow the MIT-licensed afread reference and were checked against public documents
//! saved by Affinity 1.x to 3.x (container versions 8 to 12); see the crate README for provenance
//! and the exact supported subset. Nothing here writes Affinity files.
#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

use std::fmt;

mod access;
pub mod container;
mod geometry;
pub mod model;
pub mod paint;
pub mod preview;
mod raster;
mod shapes;
pub mod stream;
#[cfg(any(test, feature = "synth"))]
pub mod synth;
mod text;

pub use container::{Archive, Limits, MAGIC, is_affinity};
pub use model::{Document, read};
pub use preview::{MAX_PREVIEW_BYTES, MAX_PREVIEW_DIMENSION, Preview, preview};

/// Why a file could not be read. Messages are short reasons; callers add advice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    Malformed(&'static str),
    Unsupported(&'static str),
    Limit(&'static str),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Malformed(s) => write!(f, "damaged Affinity file: {s}"),
            Self::Unsupported(s) => write!(f, "unsupported Affinity file: {s}"),
            Self::Limit(s) => write!(f, "Affinity file exceeds a safety limit: {s}"),
        }
    }
}

impl std::error::Error for Error {}

#[cfg(test)]
mod tests;
