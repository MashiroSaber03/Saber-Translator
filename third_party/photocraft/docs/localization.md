# UI localisation and live language switching

PhotoCraft uses the same Rust catalog implementation in the native egui shell and the
WebAssembly build. English source strings are lookup keys. UTF-8 TSV catalogs supply plain,
contextual, command-ID and plural translations; missing entries fall back to English.

## Supported languages

| Language | Preference code | Integer plural forms |
|---|---|---|
| English | `en` | one / other |
| 简体中文 | `zh-hans` | one form |
| 日本語 | `ja` | one form |
| 한국어 | `ko` | one form |
| Русский | `ru` | one / few / many |
| Українська | `uk` | one / few / many (teens use many) |
| Français | `fr` | singular for 0 and 1 / plural for 2+ |
| 繁體中文 | `zh-hant` | one form |
| Español | `es` | one / other |
| Čeština | `cs` | one / few / other |
| Deutsch | `de` | one / other |
| Bahasa Indonesia | `id` | one form |
| Português (Brasil) | `pt-br` | singular for 0 and 1 / plural for 2+ |
| Italiano | `it` | one / other |
| Polski | `pl` | one / few (2–4, not 12–14) / many |
| Ελληνικά | `el` | one / other |
| Nederlands | `nl` | one / other |
| Magyar | `hu` | one form |
| Tiếng Việt | `vi` | one form |
| Srpski | `sr` | one / few / many (teens use many) |

Languages registered with `complete_menus` cover the current menu labels, `tl!` literals, blend
mode names, and generated preference labels; tests enforce that coverage. Vietnamese (`vi`) is
selectable, resolves regional tags such as `vi-VN`, and covers all current stable localization
keys. `complete_menus` remains false while generated/dynamic long-tail labels are still under
review. This does not include every engine error or status message: those remain English, as do
automation command IDs and user-provided document, layer and preset names.

## Switch without restarting

Open **Edit > Preferences > Interface > Language** and choose a language. The Preferences
dialog previews its title, controls and buttons in the chosen language. **Cancel** discards
the preview, **Apply** commits it without closing the dialog, and **OK** commits and closes.
The rest of the interface follows the committed preference on its next draw. Existing
documents, undo history and tools remain available throughout a language change.

The existing preference store saves the selection for future launches. `auto` follows the
native system locale; an unsupported code follows the same fallback. Regional tags such as
`fr-CA`, `ko-KR`, `vi-VN` and `zh-CN` resolve to the corresponding registered catalog. Traditional and
Simplified Chinese remain distinct. Every Portuguese locale (`pt`, `pt-BR`, `pt-PT`) uses the
Brazilian Portuguese catalog. The Serbian catalog is written in Latin script (Ekavian); every
Serbian locale (`sr-RS`, `sr-Latn-RS`, `sr_RS@latin`, and Cyrillic `sr-Cyrl-RS`) resolves to it.

### First launch and system language

With no saved preference, `interface.language` defaults to `auto`, so the first rendered
window uses the first supported system UI language. An older preferences file without this
field also uses Auto. If detection fails or none of the preferred languages is registered,
the UI uses English (`en`). A saved manual language choice takes precedence on later launches;
startup never overwrites it or writes the detected language into the preference file.

The native-only, pinned `sys-locale` dependency provides safe Rust access to the platform APIs:

- **Windows:** `GetUserPreferredUILanguages`, in the user's preferred order. This uses UI
  languages rather than the Region setting for dates and numbers; they can differ.
- **macOS:** `CFLocaleCopyPreferredLanguages`, in preferred order, without starting `defaults`
  or depending on a particular `.plist` representation.
- **Linux/BSD:** the platform's standard locale environment, including `LANGUAGE` lists.

`PHOTOCRAFT_LOCALE` is an optional single-tag override for reproducible launches; an unsupported
override falls back to English. Empty overrides use the system. Unix locale variables do not
override Windows/macOS UI-language preferences. Tags are read once per process (at most 64,
128 bytes each) and matched against the supported language registry. Changes to the
OS language list itself take effect at the next launch. The web build keeps its existing
English Auto fallback until browser-locale detection is implemented.

API implementation and licensing: [sys-locale](https://github.com/1Password/sys-locale).

Agents use the existing `prefs.set` command through CLI, MCP or the desktop control channel:

```json
{"method":"engine.execute","params":{"command":"prefs.set","params":{"path":"interface.language","value":"ko"}}}
```

Control requests are processed before synchronising the drawing language, so a request
updates the labels in the same rendered frame. Drawing language is local to the UI thread;
a scoped preview restores the previous language even if drawing unwinds. Menu and dialog
identity continues to use command IDs, independent of translated labels.

This is hot switching among the bundled catalogs. Catalog files are embedded with
`include_str!`; editing a TSV file requires a rebuild. External catalog watching and
file-resource hot reload are not implemented.

## Right-to-left languages

Arabic, Hebrew, Persian, Urdu and the other right-to-left languages listed in
`i18n::dir_for_code` mirror the window's chrome when chosen: the menu bar starts at the right
with File, the toolbox docks on the right and the panel dock and its rail on the left, and the
title, options and status bars read from the right (`crates/ui-egui/src/direction.rs`). The
direction follows the language code, so a catalog needs no extra setting.

Not mirrored yet: panel interiors, dialogs, menu rows (shortcuts stay on the right), submenus
(they open to the right), document tabs and the canvas. Before an Arabic catalog reads
correctly, the UI also needs an Arabic fallback font (the bundled UI fonts have no Arabic
glyphs) and bidirectional word order: egui 0.36 shapes each right-to-left word with HarfRust,
so letters join, but lays a line's words out left to right, and its caret indexing is off in
right-to-left text (emilk/egui#8576, #8577). Document text is laid out by the text engine.

No right-to-left catalog ships yet. To check the mirrored layout with any language, launch with
`PHOTOCRAFT_UI_DIRECTION=rtl` (or `ltr` to force left-to-right); it is read once per process
and has no effect on the web build:

```sh
PHOTOCRAFT_UI_DIRECTION=rtl cargo run -p photocraft-ui-egui --example snapshot -- --out rtl.png
```

## Fonts

The native lazy CJK font loader prioritises the selected Japanese, Korean or Chinese script.
Switching languages rebuilds the font definitions and resets the existing loader, so fonts
removed from the definitions can be registered again. Idle frames keep their font cache.
Other languages retain the system's CJK fallback preference for document names and text.

Font files are not added to this repository. Native rendering uses installed system fonts
and the optional craft-fonts build input; see [Fonts](development.md#fonts-craft-fonts).
The web build can select every catalog, but CJK glyph delivery and automatic browser-locale
detection remain separate outstanding work. Switching catalogs does not supply missing fonts.

## Add or maintain a catalog

1. Add `<code>.tsv` under `crates/ui-egui/src/i18n`, following the header in `ja.tsv`.
2. Register its code, native name and plural rule in `i18n::LANGUAGES`.
3. Translate the meaning of the English source keys. Preserve `{name}` placeholders, escaped
   characters and the trailing `…` on dialog commands. Use `@id` to disambiguate a command
   label and `@plural` for count-sensitive text. Keep product names such as PhotoCraft.
4. Set `complete_menus` when the catalog covers all enforced keys. Add source and license
   attribution for translation assets, then run the catalog and shell tests.

```sh
cargo test -p photocraft-ui-egui --locked
cargo clippy -p photocraft-ui-egui --all-targets --locked -- -D warnings
cargo xtask i18n-coverage
cargo xtask layers
cargo xtask wasm
```

Render representative dialogs with the offscreen `snapshot` example, including a sequence
of language changes in one process. Inspect the PNGs for missing glyphs and truncated labels.
Translations are original work or reuse the open PhotoCraft contributions credited in
[ATTRIBUTION.md](../ATTRIBUTION.md); no proprietary localisation resources are used.
