#!/bin/sh
# Builds the Invert example plug-in and refreshes the test fixture built from it.
# Needs the wasm32-unknown-unknown target: `rustup target add wasm32-unknown-unknown`.
set -eu
cd "$(dirname "$0")"
cargo build --release --target wasm32-unknown-unknown
out=target/wasm32-unknown-unknown/release/photocraft_plugin_invert.wasm
cp "$out" ../../../crates/plugins/tests/fixtures/invert.wasm
echo "built $out ($(wc -c < "$out") bytes) -> crates/plugins/tests/fixtures/invert.wasm"
