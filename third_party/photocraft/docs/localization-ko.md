# Korean terminology and dynamic UI labels

This supplements the Korean catalog merged in #582. The existing language
registry, live language switching and Korean font loading are retained.
Translations use the meaning of PhotoCraft's English labels and terminology
observed in the Korean Photoshop UI in October 2026. No proprietary translation
resources, fonts, screenshots or user documents are included.

## Terminology

| English source | Korean UI |
|---|---|
| Gaussian Blur | 가우시안 흐림 효과 |
| Liquify | 픽셀 유동화 |
| Adaptive Wide Angle | 응용 광각 |
| Brush Tip Shape | 브러시 끝 모양 |
| Smudge Tool | 손가락 도구 |
| Puppet Warp | 퍼펫 뒤틀기 |
| Render (menu) | 렌더 |
| Sharpen (menu) | 선명 효과 |
| Sharpen (command) | 선명하게 |

Command-ID translations disambiguate menu names from actions. Photoshop terms
are used when the operation matches; PhotoCraft-specific features retain their
own meaning. Technology names, units and user-provided names are not translated.
The terminology test protects representative tool names and this distinction.

## Dynamic coverage

Generated filter parameter names, symbolic choices, gallery categories/effects,
job titles, layer-style menus and shortcut labels now go through the catalog.
The filter test enumerates engine filter/adjustment parameter specs and gallery
names to catch new Korean omissions that literal-only scans cannot find.
Two newly introduced literals, `Enable {name}` and `No patterns`, are present in
all nine non-English catalogs. Existing fallback behavior is preserved for
untranslated dynamic labels in other languages.

This is not a claim that all application output is localized. Engine error
messages, arbitrary plug-in labels and user document/layer/preset names remain
outside this translation change. The source coverage report measures its
scanned key set, not every possible runtime string.

## Validation

On main 7055f0c plus this change:

- UI tests: 728 passed, 3 ignored, including live language-switch tests.
- Clippy for the UI crate and all targets: passed with warnings denied.
- i18n coverage: 1,406/1,406 scanned keys for all ten registered languages;
  all seven coverage-tool tests passed.
- Dependency-layer and WebAssembly checks passed. Existing CMS dead-code
  warnings appear during WebAssembly checks.
- Offscreen French-to-Korean language switching and the Korean brush settings
  panel were rendered and visually inspected for missing glyphs/clipping.

Re-run with `cargo test -p photocraft-ui-egui --locked`,
`cargo clippy -p photocraft-ui-egui --all-targets --locked -- -D warnings`,
`cargo test -p xtask i18n_coverage`, `cargo xtask i18n-coverage`,
`cargo xtask layers`, and `cargo xtask wasm`.

IME composition and PSD font matching are separate contributions.
