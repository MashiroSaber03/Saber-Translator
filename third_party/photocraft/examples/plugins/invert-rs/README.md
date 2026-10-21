# Invert: an example PhotoCraft plug-in

A complete WebAssembly filter plug-in in about 70 lines of `no_std` Rust: no allocator, no imports,
under 1 KiB built. It inverts the colour channels and leaves alpha alone, matching
Image › Adjustments › Invert. The ABI is documented in [`docs/plugins.md`](../../../docs/plugins.md).

## Build

```sh
rustup target add wasm32-unknown-unknown
./build.sh
```

`build.sh` runs `cargo build --release --target wasm32-unknown-unknown` and copies the module to
`crates/plugins/tests/fixtures/invert.wasm`, the test fixture the host's tests use. This crate is not
a member of the PhotoCraft workspace.

## Install

```sh
# Into a running app (control channel) or the CLI, as a command:
plugin.install {"path": "examples/plugins/invert-rs/target/wasm32-unknown-unknown/release/photocraft_plugin_invert.wasm"}
plugin.run {"id": "org.photocraft.example.invert"}
```

Or copy the `.wasm` into the folder set in Edit › Preferences › Plug-ins (with *Use Additional
Plug-ins Folder* on); it then appears under Filter › Plug-ins.
