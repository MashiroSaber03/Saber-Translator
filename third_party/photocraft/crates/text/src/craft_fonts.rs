//! Fonts from the optional craft-fonts build input (https://github.com/storytold/craft-fonts).
//!
//! `CRAFT_FONTS` is empty unless the app was built with `CRAFT_FONTS_DIR=<craft-fonts checkout>`
//! (see `build.rs`); every user of it must work when it is empty. Today it carries the Japanese
//! fonts: BIZ UDPGothic (UI) and Shippori Mincho / BIZ UDMincho (serif document text). The web
//! build (wasm32) embeds none of them: they don't fit its size cap (see `build.rs`).

/// A font from the optional craft-fonts build input (empty unless built with `CRAFT_FONTS_DIR`).
pub struct CraftFont {
    pub family: &'static str,
    pub style: &'static str,
    /// ISO 15924 scripts the font is for, e.g. `"Jpan"`.
    pub scripts: &'static [&'static str],
    pub bytes: &'static [u8],
}

include!(concat!(env!("OUT_DIR"), "/craft_fonts.rs"));

/// The preferred UI family for Japanese.
pub const UI_JAPANESE_FAMILY: &str = "BIZ UDPGothic";

impl CraftFont {
    /// True for a font meant for Japanese text.
    pub fn is_japanese(&self) -> bool {
        self.scripts.contains(&"Jpan")
    }

    /// True for a Mincho (serif) face.
    pub fn is_mincho(&self) -> bool {
        self.family.contains("Mincho")
    }
}

/// The Japanese craft fonts in UI preference order: BIZ UDPGothic Regular first, then its other
/// styles, then the rest in manifest order.
pub fn japanese_for_ui() -> Vec<&'static CraftFont> {
    let mut v: Vec<&CraftFont> = CRAFT_FONTS.iter().filter(|f| f.is_japanese()).collect();
    v.sort_by_key(|f| (f.family != UI_JAPANESE_FAMILY, f.style != "Regular"));
    v
}

/// Japanese craft font families (each once) for document fallback: sans (Gothic) first, then
/// Mincho. Serif runs reorder them with [`mincho_first`].
pub fn japanese_families() -> Vec<&'static str> {
    let mut v: Vec<&'static str> = Vec::new();
    for f in japanese_for_ui() {
        if !v.contains(&f.family) {
            v.push(f.family);
        }
    }
    v.sort_by_key(|f| f.contains("Mincho"));
    v
}

/// `fallback` with the Japanese craft Mincho families moved ahead of the craft Gothic one, in
/// place (the CJK script order around them is unchanged). For serif runs.
pub fn mincho_first(fallback: &[String]) -> Vec<String> {
    let craft = japanese_families();
    let slots: Vec<usize> = fallback.iter().enumerate().filter(|(_, f)| craft.contains(&f.as_str())).map(|(i, _)| i).collect();
    let mut group: Vec<String> = slots.iter().filter_map(|&i| fallback.get(i).cloned()).collect();
    group.sort_by_key(|f| !f.contains("Mincho"));
    let mut out = fallback.to_vec();
    for (slot, fam) in slots.into_iter().zip(group) {
        if let Some(o) = out.get_mut(slot) {
            *o = fam;
        }
    }
    out
}

/// A family name that reads as serif (Mincho, Song/Ming, Myeongjo, Times …): its Japanese
/// fallback should be a Mincho face rather than a Gothic one.
pub fn is_serif_family(name: &str) -> bool {
    let n = name.to_ascii_lowercase();
    if n.contains("sans") || n.contains("gothic") {
        return false;
    }
    const SERIF: &[&str] = &[
        "serif",
        "mincho",
        "minchō",
        "song",
        "ming",
        "myeongjo",
        "myungjo",
        "batang",
        "times",
        "georgia",
        "garamond",
        "minion",
        "baskerville",
        "caslon",
        "bodoni",
        "didot",
        "palatino",
        "cambria",
        "book antiqua",
        "century",
        "hoefler",
    ];
    SERIF.iter().any(|s| n.contains(s))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mincho_first_reorders_only_craft_families() {
        let fams = japanese_families();
        if fams.len() < 2 {
            eprintln!("skipping: built without craft-fonts (or with only one Japanese family)");
            return;
        }
        let fallback: Vec<String> =
            ["Noto Sans"].iter().map(|s| s.to_string()).chain(fams.iter().map(|s| s.to_string())).chain(["Hiragino Sans".to_string()]).collect();
        let serif = mincho_first(&fallback);
        assert_eq!(serif.first().map(String::as_str), Some("Noto Sans"));
        assert_eq!(serif.last().map(String::as_str), Some("Hiragino Sans"));
        assert!(serif.get(1).is_some_and(|f| f.contains("Mincho")), "{serif:?}");
        assert_eq!(serif.len(), fallback.len());
    }

    #[test]
    fn mincho_first_without_craft_fonts_is_identity() {
        let fallback = vec!["Noto Sans".to_string(), "Hiragino Sans".to_string()];
        assert_eq!(mincho_first(&fallback), fallback);
    }

    #[test]
    fn serif_names() {
        for s in ["Shippori Mincho", "Times New Roman", "Noto Serif JP", "Georgia", "Songti SC", "Adobe Garamond Pro"] {
            assert!(is_serif_family(s), "{s}");
        }
        for s in ["Inter", "Noto Sans Serif", "BIZ UDPGothic", "Helvetica", "", "Noto Sans CJK JP"] {
            assert!(!is_serif_family(s), "{s}");
        }
    }

    #[test]
    fn ui_order_prefers_biz_udpgothic_regular() {
        let v = japanese_for_ui();
        let Some(first) = v.first() else {
            eprintln!("skipping: built without craft-fonts (CRAFT_FONTS is empty)");
            return;
        };
        assert_eq!((first.family, first.style), (UI_JAPANESE_FAMILY, "Regular"));
        assert!(v.iter().all(|f| f.is_japanese() && !f.bytes.is_empty()));
    }
}
