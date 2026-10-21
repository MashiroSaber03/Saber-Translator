# Releasing PhotoCraft

Every push to the `release` branch runs `.github/workflows/release.yml`. The workflow builds
signed installers for macOS, Windows, Linux and the web, plus a FreeBSD tarball, then creates
or updates a **draft** GitHub Release named `PhotoCraft v<version>`. Nobody sees a draft until a
maintainer publishes it.

This is PhotoCraft's implementation of the shared
[release playbook](release-playbook.md). User-facing names say **PhotoCraft**. Files, binaries
and ids stay lowercase (`photocraft-<version>-<platform>-<arch>.<ext>`, `ai.storyteller.photocraft`).

## Cutting a release

1. **Bump the version** on `main`. The only place it lives is `[workspace.package] version`
   in the root `Cargo.toml`:

   ```sh
   cargo xtask version                 # prints the current version, e.g. 0.1.0
   cargo xtask version set 0.2.0       # or 0.2.0-rc.1; updates Cargo.toml and Cargo.lock
   ```

   Commit the change (`Cargo.toml` + `Cargo.lock`) through the normal review flow.
2. **Merge `main` into `release`** (or fast-forward it) and push. The workflow starts by itself.
3. **Wait for the draft.** After about 30 to 45 minutes (notarization is the slow part), the
   Releases page has a draft `PhotoCraft v0.2.0`, tagged `v0.2.0` on the pushed commit, with
   every artifact and `SHA256SUMS.txt`. The notes are generated from the merged PRs.
4. **Check it.** Download an installer or two and look at the job summaries. Any
   `::warning::` there means a signing secret was missing and that artifact is unsigned.
5. **Add the scorecard deltas** to the draft's notes: what moved in
   [`docs/scorecard.md`](scorecard.md) since the previous release. Compare the summary table
   and the performance table of both tags (`git diff v<previous> v<this> -- docs/scorecard.md`)
   and list each area's done / partial / missing change, the scenarios that newly meet their
   budget or got slower, corpus floors raised, and the change in settings that do nothing.
   Quote numbers with the baseline's machine and load average.
6. **Publish** the draft in the GitHub UI. Publishing creates the `v0.2.0` tag. Versions with a
   pre-release suffix (`-rc.1`) are marked as pre-releases.

Pushing to `release` again before you publish rebuilds the same draft and replaces its assets.
After the draft is published, the workflow refuses to touch that version again, so bump it first.

**Test runs:** *Actions → Release → Run workflow* runs the whole pipeline by hand. The optional
`version` input (such as `0.2.0-rc.1`) overrides `Cargo.toml` for that run only. The jobs apply
it with `cargo xtask version set` before building, so the binaries report it too. The run still
needs the `release` environment, which only the `release` branch can use, so pick that branch
in the dialog.

## What gets built

| Platform | Artifacts | Built on |
|---|---|---|
| macOS 11+ (universal: Apple silicon + Intel) | `photocraft-<v>-macos-universal.dmg`, `photocraft-cli-<v>-macos-universal.zip` | `macos-15` |
| Windows 10+ x64 | `photocraft-<v>-windows-x64.msi`, `photocraft-<v>-windows-x64-portable.zip` | `windows-latest` |
| Windows 10+ x86 (32-bit) | `photocraft-<v>-windows-x86.msi`, `photocraft-<v>-windows-x86-portable.zip` | `windows-latest` |
| Linux x86_64 | `photocraft-<v>-linux-x86_64.{AppImage,AppImage.zsync,deb,rpm,tar.gz,flatpak}` | `ubuntu-22.04` (Flatpak: `ubuntu-24.04`) |
| Linux aarch64 | `photocraft-<v>-linux-aarch64.{AppImage,AppImage.zsync,deb,rpm,tar.gz,flatpak}` | `ubuntu-22.04-arm` (Flatpak: `ubuntu-24.04-arm`) |
| Linux riscv64 (glibc 2.39+) | `photocraft-<v>-linux-riscv64.tar.gz` | cross-compiled on `ubuntu-24.04` (ports.ubuntu.com riscv64 libs); CLI smoke-tested under QEMU |
| FreeBSD 14 x86_64 | `photocraft-<v>-freebsd-x86_64.tar.gz` | FreeBSD 14.3 VM on `ubuntu-latest` |
| Web | `photocraft-web-<v>.zip` (static site; see [`packaging/web/README.md`](../packaging/web/README.md)) | `ubuntu-latest` |

Every binary reports its version, the commit and the build date: `photocraft --version`,
`photocraft-cli --version`, and *Help › About PhotoCraft*. CI sets `PHOTOCRAFT_BUILD_SHA` and
`PHOTOCRAFT_BUILD_DATE`, and `crates/engine/src/build_info.rs` reads them at compile time. A plain
`cargo build` doesn't set them and reports `0.2.0 (dev build)`.

Every desktop build job (macOS, Windows, Linux, FreeBSD) also checks out [craft-fonts](https://github.com/storytold/craft-fonts)
at the `craft-fonts-ref` commit in `.github/workflows/release.yml` and builds with `CRAFT_FONTS_DIR` and
`CRAFT_FONTS_REQUIRED=1`, so desktop releases embed its Japanese fonts (the web build embeds none: see
`docs/development.md` › Fonts) and fail rather than ship without them. The packages carry each font's licence as
`OFL-<family>.txt` (`copy_font_licences` in `packaging/env.sh`; the portable zip on Windows;
`Contents/Resources/Licenses` in the macOS app). Bump the pin deliberately, together with the one
in `ci.yml`. Rules: `../craftrules/standards/fonts.md`; build option: `docs/development.md` › Fonts.

### macOS

`packaging/macos/package.sh` builds `aarch64-apple-darwin` and `x86_64-apple-darwin` with
`MACOSX_DEPLOYMENT_TARGET=11.0`, joins them with `lipo`, and assembles `PhotoCraft.app`:

- `Info.plist` is generated from `Info.plist.in`. The bundle id is `ai.storyteller.photocraft`.
  The plist sets `LSMinimumSystemVersion` 11.0, `NSHighResolutionCapable`, and document types:
  `.pcraft` (Owner), plus PSD/PSB and the image formats PhotoCraft reads (Alternate, so it never
  takes over Preview's defaults). The icon is `assets/app-icon/photocraft.icns`.
- **Signing** goes inside-out with the hardened runtime and a secure timestamp. The executable
  is signed first, then the bundle. There's no `--deep` on the final signature. The
  entitlements (`entitlements.plist`) are deliberately empty.
- **Notarization:** the app is zipped and sent with `xcrun notarytool submit --wait`, then the
  ticket is stapled to the app. The app goes on a DMG (`hdiutil`, with an `Applications` link
  to drag onto). The DMG is signed, notarized and stapled too. The script checks the results
  with `codesign --verify --strict`, `stapler validate` and `spctl -a -vvv`.
  Its Finder window (background, icon size and positions) comes from
  [`packaging/macos/dmg/`](../packaging/macos/dmg/README.md), and its volume is named `PhotoCraft`
  without the version, which the window's background needs; the DMG file name keeps the version.
- **CLI:** the universal `photocraft-cli` is signed with the same Developer ID, the hardened
  runtime and a secure timestamp (identifier `ai.storyteller.photocraft-cli`), zipped, and the zip
  is sent to `notarytool`. Only `.app`, `.dmg` and `.pkg` can hold a stapled ticket, not a bare
  Mach-O, so Gatekeeper looks the CLI's ticket up online the first time a downloaded
  (quarantined) copy runs. Offline, that first run can be refused until the Mac is online again.
- **Verification:** `packaging/macos/verify.sh` checks the artifacts as users download them. It
  unpacks the CLI zip and requires, for the binary inside, `codesign --verify --strict`, both
  architectures, the hardened runtime flag, a `Developer ID Application` authority from
  `APPLE_TEAM_ID`, a timestamp, and `spctl --assess --type install` reporting
  `source=Notarized Developer ID` (the same online lookup Gatekeeper does). For the DMG it runs
  `codesign --verify`, `stapler validate` and `spctl --type open`. The release workflow runs it
  as its own step after packaging. With signing secrets it fails the job on any miss; without
  them it only checks signature integrity and warns, like `package.sh`. Run it locally after a
  build: `packaging/macos/verify.sh --arch aarch64`.

Locally, without certificates, the script signs ad-hoc (`codesign -s -`) and skips notarization.
That's enough to check the bundle and the DMG on your own Mac:

```sh
packaging/macos/package.sh                    # universal; needs both rustup targets
packaging/macos/package.sh --arch aarch64     # quicker, host-only
open dist/release/photocraft-*-macos-*.dmg
```

### Windows

`packaging/windows/package.ps1 -Arch x64|x86` builds with `-C target-feature=+crt-static`.
The static C runtime means neither the MSI nor the portable zip needs the Visual C++
redistributable, which matters for a standalone installer and costs only about 100 KB. The flag
goes in `CARGO_TARGET_<TRIPLE>_RUSTFLAGS`, so host build scripts aren't affected.

- `apps/photocraft/build.rs` and `apps/photocraft-cli/build.rs` embed the icon
  (`assets/app-icon/photocraft.ico`) and VERSIONINFO with the `winresource` crate
  (build dependency `0.1`). They only do this when targeting Windows. Elsewhere each
  script is a no-op, even if `PHOTOCRAFT_REQUIRE_WINRES` is set, and the web build
  doesn't touch that crate. Both set ProductName to PhotoCraft, CompanyName to
  Learning Machines LLC and the same copyright line. The GUI description is
  "PhotoCraft image editor" (`photocraft.exe`). The CLI description is
  "PhotoCraft command-line interface", with OriginalFilename `photocraft-cli.exe`
  and InternalName `photocraft-cli`. Version strings are the Cargo package version,
  so a pre-release tag is kept; the numeric file and product versions are
  winresource's `major.minor.patch.0`. A missing resource compiler warns, so an
  optional cross-compile still links. `package.ps1` sets `PHOTOCRAFT_REQUIRE_WINRES=1`.
  The CLI build script then returns an error naming `rc.exe` or `windres` and does
  not panic.
- Release builds use the GUI subsystem, so Start Menu launches don't open a console window.
  The CLI stays console subsystem 3 so its output reaches the terminal.
- `photocraft.wxs` (WiX v5) is a per-machine install into Program Files with an advertised
  Start Menu shortcut. PhotoCraft becomes the default app for `.pcraft` and is listed under
  "Open with" for PSD/PSB and image files. It also registers App Paths (Win+R `photocraft`).
  The MSI version is the numeric `X.Y.Z`, because MSI has no pre-release field. Same-version
  upgrades are allowed so that release candidates replace each other.
- Double-clicking the MSI opens a setup wizard (WixUI_InstallDir without the licence page):
  Welcome, install folder (remembered for upgrades in `HKLM\Software\PhotoCraft\InstallDir`),
  Ready, a progress page, and a Finish page that says where to find the app, with a ticked "Launch
  PhotoCraft" box. The pages name the full version (`photocraft.en-us.wxl`). `package.ps1` draws
  the wizard's banner and side bitmaps from the app icon, so no WiX stock art ships. `msiexec /qn`
  still installs silently. Building needs the WiX UI and Util extensions:
  `wix extension add -g WixToolset.UI.wixext/5.0.2 WixToolset.Util.wixext/5.0.2`.
- Shortcut icon identifiers keep the executable's `.exe` extension: MSI uses the identifier
  as the cached icon filename, and an extensionless filename can render as a blank document
  icon. `packaging/windows/check-icons.ps1` checks the references and extensions in CI;
  `package.ps1` also validates the built MSI with ICE50 before signing it.
- **Portable zip:** it ships `packaging/windows/portable.txt` beside `photocraft.exe`. That
  marker (or a `PhotoCraft.portable` file) switches on portable mode: preferences, presets,
  recovery autosaves and the GPU startup marker go to `PhotoCraftData\` next to the exe instead
  of `%APPDATA%\Photocraft`. If that folder isn't writable the app warns and uses `%APPDATA%`.
  The MSI has no marker. The logic is in `apps/photocraft/src/app_dirs.rs` and works the same
  on macOS and Linux.
- **Signing:** `packaging/windows/sign.ps1` signs both `.exe` files and then the `.msi` with
  `signtool`, using SHA-256 and an RFC 3161 timestamp. It uses whichever material is present:
  1. a `.pfx` file (`WINDOWS_CERTIFICATE`, `WINDOWS_CERTIFICATE_PASSWORD`), timestamped by
     DigiCert, or
  2. Azure Trusted Signing (`AZURE_*`): the script downloads the
     `Microsoft.Trusted.Signing.Client` dlib and timestamps with Microsoft's server.

  The script is the single place to change when the Windows signing setup changes.

Locally on Windows: `dotnet tool install -g wix --version 5.0.2`,
`wix extension add -g WixToolset.UI.wixext/5.0.2 WixToolset.Util.wixext/5.0.2`, then
`pwsh packaging/windows/package.ps1 -Arch x64`.

### Linux

`packaging/linux/package.sh` stages one FHS tree and makes every format from it. The tree
holds both binaries, `ai.storyteller.photocraft.desktop`, hicolor icons from 16 px to 512 px
plus a scalable SVG, AppStream metainfo, and a shared-mime-info file for `.pcraft`, `.psb` and
`.qoi`.

Why these formats:

- **AppImage** runs on any distribution without installing anything. The
  download-and-go option and the fallback for distros the packages don't cover. It's built
  with the maintained `AppImage/appimagetool`, whose static runtime doesn't need libfuse2.
  The packaging script verifies pinned AppImageTool 1.9.1 and type2 runtime 20251108 binaries.
  For an offline build, pre-seed `$CARGO_TARGET_DIR/appimagetool-1.9.1-$ARCH.AppImage` and
  `$CARGO_TARGET_DIR/type2-runtime-20251108-$ARCH` with the matching official release assets;
  the script checks both SHA-256 digests and makes the cached AppImageTool executable after verification;
  the runtime is only read as data.
  An `APPIMAGETOOL` override must point to that same verified AppImageTool binary and already be
  executable (`chmod +x` it before packaging).
  Each AppImage embeds update information (`gh-releases-zsync|…|latest|…`, #349) and ships
  with a `.zsync` beside it, so AppImageUpdate, AppImageLauncher and similar tools can find
  the next release and download only the blocks that changed.
  Its `AppRun` (`packaging/linux/AppRun`) also integrates with the desktop on launch: on
  Wayland the dock icon comes from the compositor resolving the window's app_id against an
  *installed* `.desktop` file, which an AppImage alone never provides — without this every
  AppImage run shows the generic Wayland logo (#593). Best effort: it installs
  `~/.local/share/applications/ai.storyteller.photocraft.desktop` with `Exec` rewritten to
  the AppImage's path (`TryExec` dropped, the binary isn't on the host PATH) and the
  hicolor icons beside it, rewriting only when the content changed, and any failure leaves
  the launch untouched. `PHOTOCRAFT_NO_DESKTOP_INTEGRATION=1` opts out; to uninstall, `rm
  ~/.local/share/applications/ai.storyteller.photocraft.desktop` and `find
  ~/.local/share/icons/hicolor -name 'ai.storyteller.photocraft*' -delete`. The script is
  covered by `packaging/linux/apprun-test.sh` in the packaging-lint workflow.
- **.deb** covers Debian, Ubuntu, Mint, Pop!_OS and elementary. **.rpm** covers Fedora, RHEL
  and its clones, and openSUSE. Both integrate with the menu, MIME and icon caches (the
  `postinst.sh` hook) and uninstall cleanly. Both are built by
  [nfpm](https://nfpm.goreleaser.com/): one maintained, dependency-free tool and one config
  (`nfpm.yaml`). It doesn't need cargo-deb and cargo-generate-rpm metadata in our crates, and
  it can build both formats on Ubuntu.
- **.tar.gz** is for people who manage their own `/opt` or `~/.local`.
- **Flatpak bundle** (`.flatpak`) is a single file that installs into Flatpak, sandboxed and
  updatable by installing a newer bundle, for people who prefer Flatpak to AppImage.
  `packaging/linux/flatpak-bundle.sh` repackages the job's `.tar.gz` with
  `packaging/linux/flatpak/ai.storyteller.photocraft.bundle.yml` on the freedesktop 26.08
  runtime. It does no Rust build and needs no network inside `flatpak-builder`, so the bundle
  holds the same binaries as the other formats. A separate `flatpak` job per architecture
  (`ubuntu-24.04` and `ubuntu-24.04-arm`, for flatpak-builder 1.4) downloads the Linux job's
  artifact, runs the script, installs the bundle and runs `photocraft-cli --version` inside the
  sandbox as a smoke test. The bundle names Flathub as its runtime repo, so users install
  it with:

  ```sh
  flatpak install --user photocraft-<v>-linux-x86_64.flatpak   # pulls org.freedesktop.Platform//26.08 from Flathub if missing
  flatpak run ai.storyteller.photocraft
  ```

  Sandbox permissions (justified in the manifest): Wayland with X11 fallback, IPC (X11
  shared memory), `dri` for the GPU, and read/write access to Pictures and Documents. Every
  other file goes through the file-chooser and document portals. There's no network, so the
  `--control` server is only reachable from inside the sandbox until the user runs
  `flatpak override --user --share=network ai.storyteller.photocraft`. Host fonts are read from
  `/run/host/fonts` and `/run/host/user-fonts`. Locally (on Linux, with `flatpak` and
  `flatpak-builder`): `packaging/linux/package.sh --formats tar && packaging/linux/flatpak-bundle.sh`.
- **Flathub**: `packaging/linux/flatpak/ai.storyteller.photocraft.yml` builds from source and
  is ready for a Flathub submission (it keeps the same runtime and `finish-args` as the bundle
  manifest, which packaging-lint checks). CI doesn't build it. A real build needs vendored crate
  sources (`cargo-sources.json` from `flatpak-cargo-generator.py`) and adds 20+ minutes per
  architecture. Build it by hand with the commands in the manifest's header.

All Linux binaries are built on Ubuntu 22.04 and need **glibc ≥ 2.35**: Ubuntu 22.04+,
Debian 12+, Fedora 36+, RHEL 10, openSUSE Tumbleweed. They link only glibc and libgcc_s. X11,
Wayland, xkbcommon, Vulkan and EGL are loaded at runtime from the system, which is also where
the GPU driver has to come from. That's why the .deb and .rpm declare them as dependencies
(the full list, and why each is there, is in `packaging/linux/nfpm.yaml`) and the AppImage
doesn't bundle them. The Flatpak gets them from the freedesktop runtime. Because the AppImage
and the tarball can't declare dependencies, `photocraft` checks for the libraries its session
(X11 or Wayland) needs before it opens a window (`apps/photocraft/src/linux_libs.rs`) and, if
one is missing, prints the package to install and exits with status 1 instead of crashing.
`PHOTOCRAFT_SKIP_LIB_CHECK=1` skips the check.

Locally (on Linux): install [nfpm](https://nfpm.goreleaser.com/install/), then
`packaging/linux/package.sh` (or `--formats "deb tar"`).

### FreeBSD

GitHub has no FreeBSD runners, so the `freebsd` job runs `packaging/freebsd/package.sh` in a
FreeBSD 14.3 VM (`vmactions/freebsd-vm`, pinned by commit), with the same packages as the
FreeBSD CI workflow (`.github/workflows/freebsd.yml`) plus `bash`. The script builds both
binaries with `CARGO_BUILD_JOBS=4` (more runs the 12 GB VM out of memory) and writes
`photocraft-<v>-freebsd-x86_64.tar.gz`. FreeBSD's `uname -m` says `amd64`; the file name uses
`x86_64` like the other artifacts. The tarball is a `/usr/local`-style tree: `bin/photocraft`,
`bin/photocraft-cli`, and under `share/` the same desktop entry, MIME type, AppStream metainfo
and hicolor icons as Linux, plus the licences, `NOTICE` and `ATTRIBUTION.md` in
`share/doc/photocraft/`. Users install it with:

```sh
pkg install libxkbcommon wayland libX11 libXcursor libXrandr libXi libxcb mesa-libs vulkan-loader gtk3 fontconfig freetype2 alsa-lib
tar -xzf photocraft-<v>-freebsd-x86_64.tar.gz --strip-components 1 -C /usr/local
```

The job signs nothing, so it runs outside the `release` environment and gets no secrets. The
version, build date and commit reach the VM through the action's `envs:` (by name, never
templated into the script), and the build happens in `/var/tmp` so only `dist/` is copied back
out of the VM. The binaries aren't signed: FreeBSD has no code-signing scheme for loose
binaries, so check the tarball against `SHA256SUMS.txt`.

The FreeBSD CI workflow runs the same script after its build and test (on pushes to `main`, and
on PRs that touch `packaging/freebsd/`, `freebsd.yml` or `release.yml`), so the release job
isn't the first place it runs. Locally, on any OS, `packaging/freebsd/package.sh --dry-run` stages the tree from stub binaries
and lists the tarball, which checks the layout without a FreeBSD machine.

### Web

`packaging/web/package.sh` runs `trunk build --release` (see `apps/photocraft-web/Trunk.toml`)
and zips `dist/web` together with sample `_headers` and `.htaccess` files and the hosting guide.
The site only uses relative URLs, so it works under any path and in an iframe. The wasm builds
with the size-optimized `wasm-release` Cargo profile (set in `apps/photocraft-web/index.html`),
and the script fails if any `.wasm` exceeds 24 MiB, below Cloudflare's 25 MiB per-file limit.
[`packaging/web/README.md`](../packaging/web/README.md) covers MIME types, compression,
caching, the iframe snippet and the `?webgl` / `?cpu` flags.

## Secrets

All secrets live in the repository's **`release` environment** (*Settings → Environments →
release*). Restrict its deployment branches to `release`. Every job in `release.yml` that
signs or publishes declares `environment: release` (the Flatpak and FreeBSD jobs sign nothing
and don't), so only pushes to that branch, or manual runs on it, can read the secrets. Each secret is optional. If one is missing, that platform's artifacts are unsigned and
the run shows a `::warning::`.

| Secret | Used for |
|---|---|
| `APPLE_CERTIFICATE` | base64 `.p12` with "Developer ID Application: Learning Machines LLC (DJ6XS33FX8)" |
| `APPLE_CERTIFICATE_PASSWORD` | password for that `.p12` |
| `KEYCHAIN_PASSWORD` | password for the temporary CI keychain (random if unset) |
| `APPLE_ID` | Apple ID used by `notarytool` |
| `APPLE_PASSWORD` | app-specific password for that Apple ID |
| `APPLE_TEAM_ID` | `DJ6XS33FX8` |
| `WINDOWS_CERTIFICATE` | base64 `.pfx` code-signing certificate |
| `WINDOWS_CERTIFICATE_PASSWORD` | password for that `.pfx` |
| `AZURE_TENANT_ID`, `AZURE_CLIENT_ID`, `AZURE_CLIENT_SECRET` | service principal for Azure Trusted Signing (an alternative to the `.pfx`) |
| `AZURE_SIGNING_ENDPOINT` | e.g. `https://eus.codesigning.azure.net` |
| `AZURE_SIGNING_ACCOUNT`, `AZURE_CERT_PROFILE` | Trusted Signing account and certificate profile names |

`GITHUB_TOKEN` creates the release. Only the final job gets `contents: write`.

On macOS, `packaging/macos/import-cert.sh` decodes the `.p12` into a temporary keychain
(`security create-keychain`, then `security import`, then `set-key-partition-list` so codesign
can use the key without a prompt). It exports the identity's SHA-1 as `MACOS_SIGN_IDENTITY`.
The keychain is deleted at the end of the job.

## Icons

`assets/app-icon/photocraft.svg` is the canonical icon: the owner's ArtCraft drawing of a
nine-tailed kitsune, vectorised (MIT OR Apache-2.0, see `LICENSE.txt` there; palette and
geometry in `README.md`). `packaging/icons.sh` regenerates the 1024 px PNG, the `.icns`, the `.ico` (packed by
`cargo xtask ico`) and the hicolor PNGs from it. It needs `resvg`, plus `iconutil` on macOS.
The outputs are committed, so packaging never needs those tools.

## Checks

`.github/workflows/packaging-lint.yml` runs in seconds on any change to `packaging/`, the
workflows or the icons. It runs actionlint, shellcheck, a PowerShell parse, xmllint,
`desktop-file-validate`, `appstreamcli validate`, and a YAML check that the two Flatpak
manifests agree on the runtime and permissions.
