# Dependencies

Cargo dependencies are pinned through `Cargo.lock` and exercised by the normal workspace build, test, clippy, and WebAssembly jobs. Parser and GPU dependencies deserve additional review because they process attacker-controlled data or interface with native drivers.

## Current repository state

No `cargo-deny` policy file, `cargo-audit` job, or equivalent dependency-advisory workflow is present in the current tree. Do not describe dependency auditing as implemented until configuration and CI evidence land.

## Proposed checks

```sh
cargo audit
cargo deny check
```

A future `deny.toml` should be reviewed by maintainers and encode accepted licenses, advisory policy, source rules, and duplicate-version policy. Advisory exceptions need an owner, rationale, compensating control, and expiration/review date. CI tools and GitHub Actions should also be version-pinned and reviewed as dependencies.

Dependency changes should document why a crate is needed, whether it processes untrusted input, unsafe/native code exposure, supported targets, maintenance status, and license compatibility. Avoid replacing source review with a scanner result.
