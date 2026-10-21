#!/bin/sh
# Fetch every test corpus into corpus/ (pinned commits, sha256-verified; see docs/development.md,
# "Test corpora"). Same as `cargo xtask corpus --all`. Run the corpus tests with
# `cargo xtask test-corpus`.
set -eu
cd "$(dirname "$0")/.."
exec cargo xtask corpus --all "$@"
