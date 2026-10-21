# Italian terminology and dynamic UI labels

This note goes with the completion of the Italian catalog (`crates/ui-egui/src/i18n/it.tsv`).
The catalog already covered every key `cargo xtask i18n-coverage` measures, but it lacked 1,287
labels that only exist at runtime and that other catalogs (Brazilian Portuguese, Dutch, Czech)
already translate: filter and adjustment parameter names, Camera Raw panels, lighting effects,
History, Layers, path and channel menu labels, metadata fields, status messages, System Info and
dialog text. In Italian those showed in English.
They are now translated, and a review pass fixed mistranslations and inconsistent terms in the
existing entries. Translations are written from the meaning of PhotoCraft's English labels; no
proprietary translation resources, fonts, screenshots or user documents are included.

## Register and style

Commands are imperatives in the second person singular (*Elimina*, *Applica*, *Crea
pacchetto*); text addresses the user as *tu*. Only the first word is capitalised, except in
command names joined by a slash, which keep both capitals (*Luminosità/Contrasto*,
*HSL/Scala di grigio*). Decimals use a comma.

## Terminology

| English source | Italian | Note |
|---|---|---|
| Layer | Livello | |
| Canvas | Quadro | also in *Come il quadro*, *Applica ai limiti del quadro* |
| Preset | Predefinito | *Salva come predefinito* |
| Default(s) | Predefinito / Predefiniti | never *default* or *valori predefiniti* |
| Stroke | Traccia | contour and painterly strokes alike (*Tracce angolate*, *Direzione traccia*); a dash stays *Tratto* |
| Sharpen | Contrasta / Contrasto | also in Camera Raw (*Quantità contrasto*) |
| Noise | Disturbo | *Riduzione disturbo di luminanza*, *Riduzione disturbo cromatico* |
| Highlights | Luci | pairs with *Ombre* |
| Snapshot | Istantanea | |
| Smart Filters | Filtri avanzati | |
| Pattern | Pattern | *Timbro pattern* |
| Gradient stops | Punti della sfumatura | |
| Spot color | Tinta piatta | *Canale tinta piatta* |
| Click | clic | *Fai clic*, *{key}-clic*, *doppio clic* |
| Keyboard shortcut | Scorciatoia da tastiera | |
| Footage | Filmato | *Interpreta filmato*, *Sostituisci filmato* |
| Facet | Sfaccetta | |
| SemiBold / ExtraBold / ExtraLight | Semigrassetto / Extragrassetto / Extraleggero | one spelling for every weight |
| (Color) Mode | Metodo | *Cambia metodo condizionale*, never *modo* |
| Trap | Trapping | the prepress term; *Sovrastampa* is overprint, a different thing |
| HDR Toning | Mappatura tonale HDR | tone mapping; *Viraggio* stays for colour toning (*Viraggio separato*) |
| Bevel & Emboss | Smusso e rilievo | pairs with *Rilievo* (Emboss) |
| Linked (smart object) | Oggetto collegato | *Inserisci oggetto collegato*, *Converti in oggetto collegato* |

## Native review

2026-10-12: a native Italian speaker read the catalog and fixed mistranslations (Trap, HDR
Toning, Display Text, Vine Color, Override Action "Save" Commands) and unnatural phrasing (Bevel &
Emboss, East Asian / Middle Eastern Features, linked smart objects, bit depth labels).

## Not covered

Engine error and status messages, user document, layer and preset names, and automation
command IDs stay in English, as in every language. A few dynamic labels the other catalogs carry
(`Ask Before Saving Saved Embedded`, `Clear Selection`, the `Heal` family) are not used by the
current code; they are translated to stay aligned with the other catalogs.

## Validation

- `italian_covers_generated_filter_options_and_gallery_names` (`crates/ui-egui/src/filter_dialog.rs`)
  enumerates every filter and adjustment parameter, symbolic choice and gallery name, like the
  Korean test, and requires an Italian row for each. With the catalog on `main` before this change
  it lists 388 missing labels; now none.
- `cargo test -p photocraft-ui-egui --lib`: the catalog tests pass. Two layout tests,
  `type_controls_fill_the_panel_at_every_width_and_scale` and
  `properties_edit_shown_metrics_and_paragraphs_with_a_non_type_primary`, fail on `main` as well
  without this change.
- `cargo xtask i18n-coverage`: 1,726 / 1,726 for `it`.
- The offscreen `snapshot` example with `PHOTOCRAFT_LOCALE=it` rendered Lighting Effects, the
  Camera Raw filter and Wind with Italian labels, without missing glyphs or clipped text.
