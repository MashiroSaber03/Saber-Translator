# PhotoCraft Documentation

PhotoCraft is a native image editor written in Rust. This book is the maintained entry point for installation, architecture, formats, automation, development, and security documentation.

The repository's existing `docs/` directory remains authoritative for detailed design notes, parity reports, UI conventions, release procedures, and the complete control-protocol method reference. The book links to those documents instead of copying them.

## Start here

- Users and evaluators: [Installation](getting-started/installation.md)
- Contributors: [Building](getting-started/building.md) and [Contributing](getting-started/contributing.md)
- Architecture work: [Architecture overview](architecture/overview.md)
- Agents and automation clients: [Automation overview](automation/overview.md)
- Security-sensitive work: [Security overview](security/overview.md) and [Threat model](security/threat-model.md)

## Documentation for coding agents

Coding agents should read the following in order:

1. [`AGENTS.md`](https://github.com/storytold/photocraft/blob/main/AGENTS.md) for repository rules and verification gates.
2. [`README.md`](https://github.com/storytold/photocraft/blob/main/README.md) for product scope and current entry points.
3. [`book/src/architecture/`](architecture/overview.md) for crate boundaries and command flow.
4. [`book/src/automation/`](automation/overview.md) before driving the CLI, control protocol, or MCP server.
5. [`book/src/security/`](security/overview.md) before changing a parser, file I/O, automation, MCP, the control server, command execution, or serialization/deserialization.

Agents must treat document bytes, paths, command parameters, and automation messages as untrusted input. A page that describes a proposed control does not mean that control is implemented.

## Status language

Security pages use four labels consistently:

- **Implemented:** behavior confirmed in current source or tests.
- **Known limitation:** an exposed risk or missing control in current source.
- **Proposed:** a design direction that is not enforced by the current build.
- **Future hardening:** useful follow-up work without an accepted implementation yet.
