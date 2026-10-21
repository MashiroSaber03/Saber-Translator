#!/usr/bin/env bash
# Fast, network-free checks of the verified AppImage tooling cache and override.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=verified-download.sh
. "$HERE/verified-download.sh"

sha256() {
  if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi
}
fail() { echo "FAIL: $*" >&2; exit 1; }

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
source_asset="$TMP/source.AppImage"
cache="$TMP/cached.AppImage"
printf '#!/bin/sh\nexit 0\n' >"$source_asset"
expected="$(sha256 "$source_asset")"
# Setup uses the external binary because failure-path tests below shadow the chmod function.

# Offline preseed: a valid 0644 cache must not try the unavailable URL and must become runnable.
cp "$source_asset" "$cache"
/bin/chmod 644 "$cache"
download_verified "file://$TMP/unavailable" "$cache" "$expected" true || fail "offline cache was not reused"
[ -x "$cache" ] || fail "valid preseed did not become executable"

# If a valid but non-executable cache cannot be chmod'd, fail here rather than at invocation.
/bin/chmod 644 "$cache"
chmod() { return 1; }
if err="$(download_verified "file://$TMP/unavailable" "$cache" "$expected" true 2>&1)"; then fail "unfixable cache was accepted"; fi
case "$err" in *"verified asset is not executable and cannot be chmod'd: $cache"*) ;; *) fail "wrong cache error: $err" ;; esac
unset -f chmod
/bin/chmod 755 "$cache"

# A verified, executable cache can be owned by another user; do not try to chmod it again.
chmod() { fail "chmod called for an already executable cache"; }
download_verified "file://$TMP/unavailable" "$cache" "$expected" true || fail "executable cache was not reused"
unset -f chmod

# The type2 runtime is read as data; a valid 0644 preseed must remain usable as-is.
runtime="$TMP/runtime"
cp "$source_asset" "$runtime"
/bin/chmod 644 "$runtime"
chmod() { fail "chmod called for a readable runtime cache"; }
download_verified "file://$TMP/unavailable" "$runtime" "$expected" || fail "readable runtime was not reused"
unset -f chmod
[ ! -x "$runtime" ] || fail "runtime was needlessly made executable"

# A fresh runtime download is verified, readable data rather than an executable program.
fresh_runtime="$TMP/fresh-runtime"
download_verified "file://$source_asset" "$fresh_runtime" "$expected" || fail "fresh runtime download failed"
cmp -s "$source_asset" "$fresh_runtime" || fail "fresh runtime differs from verified asset"
[ -r "$fresh_runtime" ] || fail "fresh runtime is not readable"
[ ! -x "$fresh_runtime" ] || fail "fresh runtime was made executable"
mode="$(stat -c %a "$fresh_runtime" 2>/dev/null || stat -f %Lp "$fresh_runtime")"
[ "$mode" = 644 ] || fail "fresh runtime mode is $mode, expected 644"

# A corrupt cache is replaced only by bytes with the pinned digest.
printf 'corrupt' >"$cache"
download_verified "file://$source_asset" "$cache" "$expected" true || fail "corrupt cache was not replaced"
cmp -s "$source_asset" "$cache" || fail "replacement differs from verified asset"
[ -x "$cache" ] || fail "replacement is not executable"

wrong_digest="$(printf '%064d' 0)"
bad="$TMP/bad.AppImage"
if download_verified "file://$source_asset" "$bad" "$wrong_digest" >/dev/null 2>&1; then fail "wrong digest accepted"; fi
[ ! -e "$bad" ] || fail "wrong-digest download left a cache entry"

verify_appimagetool_override "$cache" "$expected" 1.9.1 x86_64 || fail "valid override rejected"
/bin/chmod 644 "$cache"
if verify_appimagetool_override "$cache" "$expected" 1.9.1 x86_64 >/dev/null 2>&1; then fail "non-executable override accepted"; fi
/bin/chmod 755 "$cache"
if verify_appimagetool_override "$cache" "$wrong_digest" 1.9.1 x86_64 >/dev/null 2>&1; then fail "wrong-digest override accepted"; fi

echo "verified AppImage tooling ok"
