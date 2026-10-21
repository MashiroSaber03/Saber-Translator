# Command engine

Every user-visible editing action is represented by a command specification in `crates/engine/src/*_cmds.rs` and registered by `crates/engine/src/commands.rs`. A specification supplies an ID, label, menu path, shortcut, parameter documentation, enabled predicate, and `run` implementation.

```text
caller
  |
  | command id + JSON parameters
  v
command registry -- enabled/parameter checks --> command run closure
                                                   |
                                                   v
                                           Session + one history step
```

The desktop UI, CLI, control protocol, and MCP tools all dispatch into this registry. Pure view/window state is the exception and is handled by the UI shell's command table.

## Input contract

Command parameters and current document state are untrusted at the command boundary. Implementations must return `Err` for missing, wrong-type, out-of-range, or contradictory parameters. Values derived from input must not reach unchecked indexing, division, allocation sizing, or panic paths.

The ignored `crates/engine/tests/panic_hunt.rs` integration test exercises commands with adversarial parameters. A changed command also needs focused graceful-failure tests and should be verified with:

```sh
cargo test -p photocraft-engine
cargo test -p photocraft-engine --test panic_hunt -- --ignored
```

The exact command-creation checklist is maintained in the existing [contribution guide](https://github.com/storytold/photocraft/blob/main/docs/contributing.md).

## Security boundary note

The registry validates command behavior, but current automation transports do not provide per-command capabilities. An MCP or control client that reaches command dispatch can request any exposed command whose runtime `enabled` predicate allows it. Capability enforcement is [proposed automation hardening](../security/automation-security.md), not an implemented control.
