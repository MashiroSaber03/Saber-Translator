# Contributing

Read [`AGENTS.md`](https://github.com/storytold/photocraft/blob/main/AGENTS.md) and the existing [contribution guide](https://github.com/storytold/photocraft/blob/main/docs/contributing.md) before editing source. Those files define the clean-room policy, crate layering, command checklist, testing requirements, naming, and visual verification rules.

Important repository invariants include:

- user-visible behavior is implemented as an engine command unless it is pure shell/view state;
- public pixel paths cannot assume 8-bit data or sRGB;
- command execution must return errors for adversarial input rather than panic;
- format work includes round-trip, malformed-input, and limit tests;
- L0-L6 changes must continue to compile for WebAssembly;
- third-party assets need a permissive license, a neighboring license file, and an `ATTRIBUTION.md` entry.

## Security-sensitive contributions

Before changing parsers, file I/O, serialization, automation, MCP, the control server, or command dispatch, read the [threat model](../security/threat-model.md) and [trust boundaries](../security/trust-boundaries.md). State whether a change adds an implemented control, documents a limitation, or proposes later hardening. Add a regression test for every fixed security defect when a safe test is possible.

Do not include live exploit payloads, secrets, personal files, or unlicensed samples in issues, commits, fixtures, or fuzz corpora. Follow [Vulnerability reporting](../security/vulnerability-reporting.md) for undisclosed vulnerabilities.
