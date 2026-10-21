//! Sandboxed WebAssembly plug-ins for PhotoCraft.
//!
//! A plug-in is a WebAssembly module that implements the PhotoCraft plug-in ABI (v1, documented
//! in `docs/plugins.md`). It runs in [`wasmi`], a pure-Rust interpreter, with:
//!
//! - **no host imports**: no WASI, file system, network, clock or randomness. A module that
//!   imports anything is rejected;
//! - an **instruction budget** (fuel) per call, metered in slices so a wall-clock deadline is
//!   also enforced (native builds), so infinite loops end with an error;
//! - a **linear-memory cap**, a **recursion-depth cap** and wasmi's strict module limits;
//! - a **fresh instance per band**, so no state leaks between calls.
//!
//! Every failure (malformed module, trap, exhausted budget, bad output) is an [`Error`]; nothing
//! a module does can panic or hang the host.
//!
//! Pixels are handed over as interleaved, normalised `f32` samples in the document's own colour
//! model (bit-depth and colour-mode agnostic), in horizontal bands so the copy stays bounded.
//! [`Plugin::apply`] runs a filter on a [`photocraft_raster::Surface`], blending by the selection.
//! The process-wide [`registry`] holds the installed plug-ins.
#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

mod apply;
pub mod manifest;
pub mod registry;
mod runtime;

pub use apply::format_code;
pub use manifest::{Area, Kind, Manifest, ParamSpec};
pub use runtime::{ABI_VERSION, Limits, Plugin};

/// A plug-in error. Every way a module can misbehave ends here, never in a panic.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum Error {
    #[error("not a valid plug-in module: {0}")]
    Module(String),
    #[error("plug-in ABI: {0}")]
    Abi(String),
    #[error("plug-in manifest: {0}")]
    Manifest(String),
    #[error("plug-in trapped: {0}")]
    Trap(String),
    #[error("plug-in exceeded its {0}")]
    Limit(String),
    #[error("plug-in parameters: {0}")]
    Params(String),
    #[error("plug-in failed: {0}")]
    Failed(String),
    #[error("no plug-in {0:?} is installed")]
    NotFound(String),
    #[error("{0}")]
    Io(String),
}

pub type Result<T> = std::result::Result<T, Error>;
