# Threat model

This threat model covers the source repository's desktop, CLI, WebAssembly, format, and automation surfaces. It assumes files, paths, JSON, MCP messages, agent decisions, and embedded metadata may be malicious.

## Assets

- confidentiality of user documents, metadata, credentials, and unrelated local files;
- integrity of documents, preferences, filesystem contents, and application state;
- availability of PhotoCraft and the host system;
- integrity and provenance of release artifacts and dependencies;
- predictable command execution and document fidelity.

## Threat actors

- an author of a malicious image, PSD/PSB, `.pcraft`, profile, LUT, pattern, or embedded object;
- a malicious or compromised local process connecting to a loopback server;
- a malicious, compromised, or over-privileged MCP client or automation agent;
- a project/action file that induces unsafe paths or expensive commands;
- a compromised dependency or build/release environment.

## Entry points

```text
untrusted files and bytes
  PSD / PSB / raster / .pcraft / ICC / LUT / pattern / embedded content

untrusted control data
  CLI arguments / filesystem paths / JSON-lines / MCP requests / agent commands

untrusted build inputs
  dependencies / CI actions / release secrets / packaging tools
```

## Primary threat scenarios

### Hostile document parsing

A compact file declares oversized dimensions, excessive channels, nested descriptors, corrupted offsets, or highly expanding compressed data. Desired outcome: reject it before unreasonable allocation or work, return an error, and retain a minimized regression test.

### Filesystem authority abuse

An automation client supplies an absolute path, traversal sequence, symlink, junction, or replaced path to read or overwrite data outside the intended project. Remote filesystem methods use separately configured directory capabilities, and engine commands that still require ambient paths fail closed at the automation boundary. This is not yet a general per-command capability model, and trusted desktop operations can still use ambient paths.

### Local control takeover

Another process on the same host connects to the loopback control server, discovers methods, drives UI input, executes commands, or opens/saves files. A 256-bit bearer token is required before method dispatch, but authentication does not grant per-client or per-tool capabilities.

### Resource exhaustion

A client opens many TCP connections, sends an extremely long JSON line, creates a huge batch, requests full-size rendering, or repeatedly invokes expensive commands. Request-line, connection, batch, and preview limits mitigate some cases; explicit JSON-depth, document-memory, and command-duration budgets remain absent.

### Supply-chain compromise

A vulnerable or malicious crate, action, installer tool, or signing environment changes build behavior or artifacts. Lockfiles and release checksums help reproducibility and verification, but advisory/license policy checks and SBOM output are future work.

## Security objectives

- reject malformed and oversized data before allocation where possible;
- use checked arithmetic and bounded recursion at parser/command boundaries;
- make automation authority explicit, narrow, and revocable;
- keep file access inside authorized roots despite traversal and link tricks;
- bound work per request and per session;
- preserve sufficient non-secret audit data to investigate automation actions;
- turn every fixed security bug into a permanent regression test.

## Non-goals and limits

PhotoCraft is not currently a sandbox for hostile automation or hostile native code. Loopback transport, Rust memory safety, typed schemas, and file-format checks do not replace authentication, authorization, OS isolation, or operational least privilege. This document does not claim those missing controls are present.
