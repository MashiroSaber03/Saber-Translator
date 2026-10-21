# Automation security model

Automation requests are untrusted input with potential effects beyond the active document. Opening, saving, rendering to a path, preferences changes, application control, pointer/type injection, and arbitrary engine commands have different risk levels but currently share broad transport access.

| Control | Current status |
|---|---|
| Loopback-only TCP binding | **Implemented** for desktop control and headless TCP serve |
| Stdio transport | **Implemented** for MCP and headless JSON lines |
| Command-level error handling | **Implemented**, with command-specific tests and `panic_hunt` coverage |
| Control-channel authentication | **Implemented:** 256-bit bearer token required before TCP method dispatch |
| MCP capability scopes | **Partial:** filesystem read/write capabilities only; general tool scopes are absent |
| Allowed read/write roots | **Implemented:** separate launch-time roots, with absent authority failing closed |
| Symlink-safe capability filesystem | **Implemented:** relative operations use held directory capabilities and reject link escapes |
| Request-byte and JSON-depth limits | **Partial:** 1 MiB request-line limit on TCP and headless JSON-lines stdio (MCP stdio input is not byte-capped); no explicit JSON-depth policy |
| Batch-step limit | **Implemented:** 256 steps for headless and MCP batches |
| Reply-size limit | **Implemented:** 8 MiB encoded JSON/MCP result; aggregate batch reply budget |
| Headless preview budgets | **Implemented:** 2048-pixel requested edge, 67,108,864 source pixels, 5 MiB encoded PNG |
| Connection/worker limit | **Implemented:** 16 active TCP connections; one worker thread per accepted active connection |
| Security audit events | **Proposed** |

## Gateway status and remaining work

```text
MCP or control client
          |
 authenticated session
          v
 security gateway
   - general method capabilities (proposed)
   - path handles/read-write roots (implemented)
   - request budgets
   - command policy
   - audit events
          |
          v
 command engine / UI shell
```

The current token proves possession of a secret but does not provide general method authorization.
Filesystem handles and a defensive command-path policy are implemented; explicit non-filesystem
capabilities, session-memory/command-duration budgets, and audit events remain. Preview and
reply ceilings do not bound all document operations or desktop screenshot capture. Private token handling, process
isolation, and least-privileged execution are still practical containment measures.
