# Fuzzing

Fuzzing procedures, current targets, security properties, and corpus rules are maintained in [Fuzzing and regression testing](../security/fuzzing.md).

The existing fuzz workspaces are intentionally separate from the main Cargo workspace because `cargo-fuzz` uses nightly Rust and sanitizer instrumentation. Example:

```sh
rustup toolchain install nightly
cargo install cargo-fuzz --locked
cd crates/codecs
cargo +nightly fuzz run decode_any
```

Reproduce a crash, minimize it, add a deterministic regression test, and then fix it. A time-limited fuzz run with no findings is evidence of that run only; it is not proof that a parser is secure.

The main GitHub Actions workflow currently runs fuzzing only when manually dispatched with the `fuzz` input. Scheduled or hosted continuous fuzzing remains future work.
