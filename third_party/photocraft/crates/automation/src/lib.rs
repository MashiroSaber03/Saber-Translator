//! # photocraft-automation
//!
//! Agent-facing automation (architecture §12):
//!
//! * [`PhotocraftMcp`]: an MCP server (official `rmcp` SDK) exposing session,
//!   document and command tools. Headless mode drives an in-process
//!   [`photocraft_engine::Session`]; bridge mode forwards to a running desktop
//!   app over the JSON-lines control protocol (`docs/control-protocol.md`), so
//!   agents can also inspect, screenshot and click the live UI.
//! * [`Headless`]: the synchronous session + file I/O core, shared with the CLI.
//! * [`rpc`]: a headless JSON-lines server (stdio or loopback TCP) with the
//!   control protocol's envelope, used by `photocraft-cli serve`.
//! * [`files`]: open/save any supported format, `.pcraft` natively.
//!
//! L6, no UI-toolkit dependencies.
#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

pub mod bridge;
pub mod budgets;
pub mod files;
pub mod headless;
pub mod rpc;
pub mod security;
pub mod server;
pub mod workspace;

pub use bridge::BridgeClient;
pub use headless::Headless;
pub use server::{Backend, PhotocraftMcp};
pub use workspace::AuthorizedWorkspace;

#[derive(Debug, thiserror::Error)]
pub enum AutomationError {
    #[error("{0}")]
    BadRequest(String),
    #[error("I/O: {0}")]
    Io(String),
    #[error(transparent)]
    Engine(#[from] photocraft_engine::EngineError),
    #[error(transparent)]
    Import(#[from] photocraft_io::IoError),
    #[error(transparent)]
    Format(#[from] photocraft_format::FormatError),
    #[error("bridge: {0}")]
    Bridge(String),
    #[error("app: {0}")]
    App(String),
    #[error("{0}")]
    Other(String),
}
