#!/usr/bin/env bash
# Smoke test for packaging/linux/AppRun: the AppImage's desktop integration writes
# and refreshes the host .desktop entry and icons, is idempotent, honours the opt-out
# and an unwritable HOME, and always launches the app (#593).
#
# Runs in seconds with no build (CI: .github/workflows/packaging-lint.yml).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
APP_ID=ai.storyteller.photocraft

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# Fake AppDir with the real AppRun, .desktop template, icons, and a stub binary.
APPDIR="$TMP/PhotoCraft.AppDir"
mkdir -p "$APPDIR/usr/bin"
install -Dm755 "$HERE/AppRun" "$APPDIR/AppRun"
install -Dm644 "$HERE/$APP_ID.desktop" "$APPDIR/usr/share/applications/$APP_ID.desktop"
mkdir -p "$APPDIR/usr/share/icons"
cp -R "$ROOT/assets/app-icon/hicolor" "$APPDIR/usr/share/icons/"
printf '#!/bin/sh\necho "stub: $*"\n' >"$APPDIR/usr/bin/photocraft"
chmod +x "$APPDIR/usr/bin/photocraft"

APPIMAGE="$TMP/photocraft test.AppImage" # spaces on purpose: Exec must stay quoted
: >"$APPIMAGE"
export APPIMAGE
export HOME="$TMP/home"
export XDG_DATA_HOME="$HOME/.local/share"
mkdir -p "$HOME"

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

# 1. Launch works and passes arguments through.
out="$("$APPDIR/AppRun" open file.png)" || fail "AppRun exited non-zero"
[ "$out" = "stub: open file.png" ] || fail "arguments not forwarded: $out"

DESKTOP="$XDG_DATA_HOME/applications/$APP_ID.desktop"
[ -f "$DESKTOP" ] || fail "desktop entry not installed"

# 2. Exec rewritten to the AppImage (quoted, spaces intact), TryExec dropped,
#    everything else preserved from the packaged entry.
expected_exec="Exec=\"$(readlink -f "$APPIMAGE")\" %F"
grep -Fxq "$expected_exec" "$DESKTOP" || fail "Exec not rewritten: $(grep '^Exec=' "$DESKTOP")"
grep -q '^TryExec=' "$DESKTOP" && fail "TryExec kept, the entry would be hidden"
grep -Fxq "Icon=$APP_ID" "$DESKTOP" || fail "Icon line lost"
grep -Fxq "X-AppImage-Integrate=false" "$DESKTOP" || fail "X-AppImage-Integrate=false missing (#3623)"
grep -Fxq 'Categories=Graphics;2DGraphics;RasterGraphics;Photography;' "$DESKTOP" || fail "metadata lines lost"
if command -v desktop-file-validate >/dev/null; then
  desktop-file-validate "$DESKTOP" || fail "generated desktop entry is invalid"
fi

# 3. Icons landed under the name the .desktop entry references.
for f in "16x16/apps/$APP_ID.png" "256x256/apps/$APP_ID.png" "scalable/apps/$APP_ID.svg"; do
  [ -f "$XDG_DATA_HOME/icons/hicolor/$f" ] || fail "icon missing: $f"
done

# 4. Idempotent: a second launch must not rewrite anything (inode unchanged).
before="$(stat -c %i "$DESKTOP")"
out="$("$APPDIR/AppRun")" || fail "second launch exited non-zero"
[ "$out" = "stub: " ] || fail "second launch output: $out"
after="$(stat -c %i "$DESKTOP")"
[ "$before" = "$after" ] || fail "desktop entry rewritten although unchanged"

# 5. Moving the AppImage refreshes Exec.
mv "$APPIMAGE" "$TMP/moved.AppImage"
export APPIMAGE="$TMP/moved.AppImage"
"$APPDIR/AppRun" >/dev/null || fail "launch after move exited non-zero"
grep -Fxq "Exec=\"$(readlink -f "$APPIMAGE")\" %F" "$DESKTOP" || fail "Exec not refreshed after move"

# 6. External AppImage integrator (#3623): if another .desktop file in $XDG_DATA_HOME/applications
#    already launches this AppImage (e.g. AppImageLauncher's appimagekit-*.desktop), AppRun removes
#    its own duplicate .desktop entry while keeping the hicolor icons installed.
ext_desktop="$XDG_DATA_HOME/applications/appimagekit-photocraft.desktop"
printf '[Desktop Entry]\nType=Application\nName=PhotoCraft\nExec="%s" %%F\nIcon=%s\n' "$(readlink -f "$APPIMAGE")" "$APP_ID" >"$ext_desktop"
"$APPDIR/AppRun" >/dev/null || fail "launch with external integrator desktop entry exited non-zero"
[ ! -e "$DESKTOP" ] || fail "duplicate desktop entry kept alongside external integrator entry"
[ -f "$ext_desktop" ] || fail "external integrator desktop entry removed"
[ -f "$XDG_DATA_HOME/icons/hicolor/256x256/apps/$APP_ID.png" ] || fail "icons missing with external integrator"
rm -f "$ext_desktop"

# 7. DESKTOPINTEGRATION set by an external integrator skips creating a second .desktop entry.
rm -rf "$XDG_DATA_HOME"
DESKTOPINTEGRATION=AppImageLauncher "$APPDIR/AppRun" >/dev/null || fail "DESKTOPINTEGRATION launch exited non-zero"
[ ! -e "$DESKTOP" ] || fail "DESKTOPINTEGRATION still wrote duplicate desktop entry"
[ -f "$XDG_DATA_HOME/icons/hicolor/256x256/apps/$APP_ID.png" ] || fail "icons not installed under DESKTOPINTEGRATION"

# 8. Opt-out installs nothing.
rm -rf "$XDG_DATA_HOME"
PHOTOCRAFT_NO_DESKTOP_INTEGRATION=1 "$APPDIR/AppRun" >/dev/null || fail "opt-out launch exited non-zero"
[ ! -e "$XDG_DATA_HOME" ] || fail "opt-out still integrated"

# 9. An unwritable data dir never blocks the launch.
mkdir -p "$TMP/ro" && chmod a-w "$TMP/ro"
out="$(XDG_DATA_HOME="$TMP/ro/data" "$APPDIR/AppRun" out.txt)" || fail "launch blocked by unwritable data dir"
[ "$out" = "stub: out.txt" ] || fail "stub not run against unwritable data dir: $out"
chmod u+w "$TMP/ro"

# 10. No $APPIMAGE (extracted AppDir): launch only, nothing installed.
rm -rf "$XDG_DATA_HOME"
env -u APPIMAGE "$APPDIR/AppRun" >/dev/null || fail "extracted-AppDir launch exited non-zero"
[ ! -e "$XDG_DATA_HOME" ] || fail "integrated without $APPIMAGE"

# 11. A path that would need Exec escaping (here a `$`): launch only, nothing installed.
rm -rf "$XDG_DATA_HOME"
odd="$TMP/odd\$(id).AppImage"
: >"$odd"
out="$(APPIMAGE="$odd" "$APPDIR/AppRun" x)" || fail "launch with an odd path exited non-zero"
[ "$out" = "stub: x" ] || fail "odd path: $out"
[ ! -e "$XDG_DATA_HOME" ] || fail "integrated an AppImage path that needs escaping"

echo "AppRun desktop integration ok"
