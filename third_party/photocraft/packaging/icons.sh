#!/usr/bin/env bash
# Regenerate every app icon from assets/app-icon/photocraft.svg (the canonical master).
#
# Needs: resvg (brew install resvg / cargo install resvg). On macOS, iconutil also writes the
# .icns. The outputs are committed, so packaging never needs these tools.
#
#   packaging/icons.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIR="$ROOT/assets/app-icon"
SVG="$DIR/photocraft.svg"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

command -v resvg >/dev/null || { echo "error: resvg not found (brew install resvg)" >&2; exit 1; }

# The artwork is a full-bleed 512-unit tile (rx=112). macOS icons pad it to Apple's 824/1024 body
# grid (transparent margin). Windows and Linux icons crop 22 units off each side (into the
# rounded corners) so the portrait reads at 16-48 px.
grep -q 'viewBox="0 0 512 512"' "$SVG" || { echo "error: expected viewBox=\"0 0 512 512\" in $SVG" >&2; exit 1; }
MAC="$TMP/mac.svg"
sed 's/viewBox="0 0 512 512"/viewBox="-62 -62 636 636"/' "$SVG" >"$MAC"
TIGHT="$TMP/tight.svg"
sed 's/viewBox="0 0 512 512"/viewBox="22 22 468 468"/' "$SVG" >"$TIGHT"

render() { resvg -w "$2" -h "$2" "$1" "$3" </dev/null; }

render "$MAC" 1024 "$DIR/photocraft-1024.png"

# Linux hicolor theme.
for s in 16 24 32 48 64 128 256 512; do
  mkdir -p "$DIR/hicolor/${s}x${s}/apps"
  render "$TIGHT" "$s" "$DIR/hicolor/${s}x${s}/apps/ai.storyteller.photocraft.png"
done
mkdir -p "$DIR/hicolor/scalable/apps"
# The lighter trace (photocraft-small.svg) keeps the scalable theme icon cheap to render.
cp "$DIR/photocraft-small.svg" "$DIR/hicolor/scalable/apps/ai.storyteller.photocraft.svg"

# Windows .ico.
ICO_PNGS=()
for s in 16 20 24 32 40 48 64 128 256; do
  render "$TIGHT" "$s" "$TMP/ico-$s.png"
  ICO_PNGS+=("$TMP/ico-$s.png")
done
(cd "$ROOT" && cargo run -q -p xtask -- ico "$DIR/photocraft.ico" "${ICO_PNGS[@]}")

# macOS .icns.
if command -v iconutil >/dev/null; then
  SET="$TMP/photocraft.iconset"
  mkdir -p "$SET"
  for s in 16 32 128 256 512; do
    render "$MAC" "$s" "$SET/icon_${s}x${s}.png"
    render "$MAC" $((s * 2)) "$SET/icon_${s}x${s}@2x.png"
  done
  iconutil -c icns -o "$DIR/photocraft.icns" "$SET"
else
  echo "warning: iconutil not found (macOS only); photocraft.icns not regenerated" >&2
fi
echo "icons written to $DIR"
