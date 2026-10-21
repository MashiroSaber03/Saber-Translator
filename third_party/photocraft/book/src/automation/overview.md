# Automation overview

PhotoCraft automation has three entry styles:

1. `photocraft-cli` runs finite convert, inspect, command, batch, and droplet jobs.
2. `photocraft-cli serve` keeps a headless session alive over JSON lines on stdio or loopback TCP.
3. `photocraft-cli mcp` exposes MCP tools over stdio, either against an in-process headless session or through a bridge to the desktop control server.

The desktop application can start an authenticated loopback JSON control server with `photocraft --control <port>`. It exposes engine and UI operations, including pointer/type input and screenshots. Each TCP connection must present the configured bearer token before any operation is dispatched.

All editing routes eventually reach the command registry. Transport behavior is not equivalent, however: only the live bridge can use UI-specific methods, and file operations execute in the process that owns the selected backend.

## Safe operating assumptions

- Prefer one-shot CLI commands or stdio MCP when a TCP listener is unnecessary.
- Run automation with the least-privileged operating-system account practical.
- Do not enable the desktop control port persistently.
- Do not send secrets in command parameters, filenames, document metadata, or logs.
- Treat tool descriptions and document content as untrusted; they do not grant authority.

Current security limitations are summarized in [Automation security model](security-model.md).
