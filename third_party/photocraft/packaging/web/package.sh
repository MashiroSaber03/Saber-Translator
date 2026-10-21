#!/usr/bin/env bash
# Build the browser version and zip it:  $DIST/photocraft-web-<version>.zip
#
# Usage: packaging/web/package.sh [--skip-build]
#
# Needs: trunk (brew install trunk / cargo install trunk --locked) and the wasm32-unknown-unknown
# target. The zip holds a self-contained static site in photocraft-web-<version>/ that works
# from any URL path and inside an <iframe>. Hosting notes: packaging/web/README.md.
set -euo pipefail
# shellcheck source=../env.sh
. "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
HERE="$ROOT/packaging/web"

if [ "${1:-}" != "--skip-build" ]; then
  command -v trunk >/dev/null || { echo "error: trunk not found (cargo install trunk --locked)" >&2; exit 1; }
  (cd "$ROOT/apps/photocraft-web" && trunk build --release)
fi

SITE="$ROOT/dist/web"
[ -f "$SITE/index.html" ] || { echo "error: $SITE/index.html missing; run without --skip-build" >&2; exit 1; }
# Paths must be relative so the site works under any prefix (public_url = "./" in Trunk.toml).
if grep -Eq '(src|href)="/[^/]' "$SITE/index.html"; then
  echo "error: $SITE/index.html has root-absolute URLs; it would break when served from a sub-path" >&2
  exit 1
fi

# Size gate (issue #198): Cloudflare Pages/Workers reject any single file over 25 MiB, and other
# hosts and CDNs have similar caps. Fail well before that so a regression shows up here, not at
# upload time. Override with PHOTOCRAFT_WASM_MAX_BYTES only to investigate.
MAX_WASM_BYTES="${PHOTOCRAFT_WASM_MAX_BYTES:-25165824}" # 24 MiB
WASM_COUNT=0
for wasm in "$SITE"/*.wasm; do
  [ -f "$wasm" ] || continue
  WASM_COUNT=$((WASM_COUNT + 1))
  size=$(wc -c <"$wasm" | tr -d ' ')
  echo "$(basename "$wasm"): $size bytes ($((size / 1048576)) MiB; limit $MAX_WASM_BYTES)"
  if [ "$size" -gt "$MAX_WASM_BYTES" ]; then
    echo "error: $(basename "$wasm") is $size bytes, over the $MAX_WASM_BYTES-byte limit (Cloudflare's per-file cap is 25 MiB)" >&2
    exit 1
  fi
done
[ "$WASM_COUNT" -gt 0 ] || { echo "error: no .wasm in $SITE" >&2; exit 1; }

NAME="photocraft-web-$VERSION"
WORK="$CARGO_TARGET_DIR/web-package"
rm -rf "$WORK"
mkdir -p "$WORK/$NAME"
cp -R "$SITE/." "$WORK/$NAME/"
# Sample server configs (MIME type, caching, compression); harmless where unused.
cp "$HERE/_headers" "$HERE/.htaccess" "$WORK/$NAME/"
cp "$HERE/README.md" "$WORK/$NAME/HOSTING.md"
copy_docs "$WORK/$NAME"
# Optional served fonts (README.md › Fonts): PHOTOCRAFT_WEB_FONTS_DIR=<a craft-fonts checkout, or any
# folder with a fonts/manifest.txt in its format> copies the manifest, every font and licence file
# it lists into the site, at the same paths. The wasm is unchanged; the app fetches fonts on demand.
if [ -n "${PHOTOCRAFT_WEB_FONTS_DIR:-}" ]; then
  FONTS_MANIFEST="$PHOTOCRAFT_WEB_FONTS_DIR/fonts/manifest.txt"
  [ -f "$FONTS_MANIFEST" ] || { echo "error: $FONTS_MANIFEST missing (PHOTOCRAFT_WEB_FONTS_DIR)" >&2; exit 1; }
  mkdir -p "$WORK/$NAME/fonts"
  cp "$FONTS_MANIFEST" "$WORK/$NAME/fonts/manifest.txt"
  # Field 3 is the font file, field 6 (if any) its licence file; both relative to the folder.
  awk -F' [|] ' '!/^[[:space:]]*(#|$)/ { print $3; if ($6 != "") print $6 }' "$FONTS_MANIFEST" | sort -u | while IFS= read -r f; do
    case "$f" in
      "" | /* | *..* | *\\* | *://*) echo "error: $FONTS_MANIFEST: '$f' is not a relative path inside the folder" >&2; exit 1 ;;
    esac
    mkdir -p "$WORK/$NAME/$(dirname "$f")"
    cp "$PHOTOCRAFT_WEB_FONTS_DIR/$f" "$WORK/$NAME/$f"
  done
  echo "served fonts: $(grep -cvE '^[[:space:]]*(#|$)' "$FONTS_MANIFEST") manifest lines from $PHOTOCRAFT_WEB_FONTS_DIR"
fi
rm -f "$DIST/$NAME.zip"
(cd "$WORK" && zip -qr9 "$DIST/$NAME.zip" "$NAME")
echo "wrote $DIST/$NAME.zip"
ls -lh "$DIST/$NAME.zip"
