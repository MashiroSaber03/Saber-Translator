# Control protocol

The desktop control protocol is newline-delimited JSON over a TCP listener bound to `127.0.0.1`. The first frame on every connection must be `auth` with the configured 256-bit bearer token. After authentication, each request carries `id`, `method`, and optional `params`; each reply includes the matching `id` and either a result or error.

```json
{"id":"auth","method":"auth","params":{"token":"<64-hex>"}}
{"id":1,"method":"ui.inspect","params":{}}
```

The current method catalog includes engine execution and discovery, application open/save operations, UI inspection and input, screenshots, and document/control helpers. The authoritative method and parameter tables are maintained in [`docs/control-protocol.md`](https://github.com/storytold/photocraft/blob/main/docs/control-protocol.md).

Implementation is split across:

- `apps/photocraft/src/control_server.rs`: loopback TCP transport;
- `crates/ui-egui/src/control.rs`: live application handlers;
- `crates/automation/src/bridge.rs`: MCP-to-GUI client.

## Current security posture

**Implemented:** the desktop server binds to IPv4 loopback; every TCP connection authenticates
before dispatch; the bridge accepts only loopback-style addresses and authenticates on
connection; encoded request lines are limited to 1 MiB; active connections are limited to 16;
socket I/O and handler waits use timeouts; MCP and headless batches are limited to 256 steps.
JSON-lines replies and encoded MCP tool results are capped at 8 MiB, with aggregate batch reply
budgets. Headless previews have source-pixel, requested-edge, and encoded-PNG ceilings. The
bridge bounds incoming reply frames and screenshot decoding.
Automation file methods require separately configured read and write roots. Request paths are
relative, and absolute, traversal, alternate-separator, device-name, and link-escape paths are
rejected before file effects.

**Known limitations:** the token is a bearer credential and grants the complete non-filesystem
control surface. There is no encryption, client identity, general per-method capability check,
explicit JSON-depth policy, aggregate session-memory accounting, or command-duration/cancellation
budget. Desktop capture/encoding and import/export still need their own operation budgets.
The listener creates one thread per authenticated
active connection, within the connection cap. Trusted one-shot CLI operations and interactive
desktop file pickers retain the launching user's normal filesystem authority.

Use a private token file, enable `--control` only for a trusted local automation session, and never proxy the unencrypted protocol beyond loopback. See [Automation security](../security/automation-security.md) for the remaining gateway work.
