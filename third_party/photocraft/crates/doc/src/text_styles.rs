//! Named type styles (Photoshop's Character Styles and Paragraph Styles panels).
//!
//! A document owns a list of **character styles** (partial character attributes) and
//! **paragraph styles** (partial paragraph *and* character attributes). "None" (no character
//! style) and "Basic Paragraph" (the default paragraph style, which can be redefined but not
//! deleted) are implicit.
//!
//! Text keeps fully resolved attributes in its runs (so layout never needs the style list). A run
//! references its styles through [`CharStyle::style_sheet`] (character style id) and the
//! paragraph's [`ParagraphStyle::style_sheet`] (paragraph style id, `None` = Basic Paragraph).
//! The attributes a run would have from its styles alone are
//! `defaults ← Basic Paragraph ← paragraph style ← character style`; whatever differs from that
//! is a **local override**, shown as "+" after the style name, as in Photoshop.
//!
//! Attribute sets are JSON objects keyed by the serde field names of [`CharStyle`] /
//! [`ParagraphStyle`] (`size_pt`, `font_family`, `align`, …), so a set can name any subset of
//! the fields and new fields need no extra code.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::text::{CharStyle, ParagraphStyle};

/// A partial attribute set: serde field name → value.
pub type StyleAttrs = Map<String, Value>;

/// The key that holds a run's style reference; never part of an attribute set.
const REF_KEY: &str = "style_sheet";

/// A named character style.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct CharacterStyleDef {
    /// Stable id (≥ 1), referenced by [`CharStyle::style_sheet`].
    pub id: u32,
    pub name: String,
    /// Character attributes this style sets (others come from the paragraph style).
    pub attrs: StyleAttrs,
}

/// A named paragraph style (or Basic Paragraph).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ParagraphStyleDef {
    /// Stable id (≥ 1; Basic Paragraph is 0), referenced by [`ParagraphStyle::style_sheet`].
    pub id: u32,
    pub name: String,
    /// Paragraph attributes this style sets.
    pub para_attrs: StyleAttrs,
    /// Character attributes this style sets (the basis of every run in its paragraphs).
    pub char_attrs: StyleAttrs,
}

/// The document's named type styles.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TextStyles {
    /// Character styles in panel order ("None" is implicit and first).
    pub character: Vec<CharacterStyleDef>,
    /// Paragraph styles in panel order (Basic Paragraph is `basic`, listed first).
    pub paragraph: Vec<ParagraphStyleDef>,
    /// Basic Paragraph (id 0). Empty attribute sets = the defaults.
    pub basic: ParagraphStyleDef,
    /// Style sheets of the PSD text engine this document was read from (`ResourceDict`
    /// `StyleSheetSet` / `ParagraphSheetSet`), kept so unmapped data survives a round trip.
    pub psd_raw: Option<Value>,
}

/// Name of the implicit default paragraph style.
pub const BASIC_PARAGRAPH: &str = "Basic Paragraph";
/// Name of the implicit "no character style" entry.
pub const NO_CHARACTER_STYLE: &str = "None";

impl Default for TextStyles {
    fn default() -> Self {
        Self {
            character: Vec::new(),
            paragraph: Vec::new(),
            basic: ParagraphStyleDef { id: 0, name: BASIC_PARAGRAPH.into(), ..Default::default() },
            psd_raw: None,
        }
    }
}

/// Serializes a style to its JSON object (without the style reference).
fn object<T: Serialize>(s: &T) -> StyleAttrs {
    match serde_json::to_value(s) {
        Ok(Value::Object(mut m)) => {
            m.remove(REF_KEY);
            m
        }
        _ => Map::new(),
    }
}

/// `base` with `attrs` applied. Fails if an attribute has the wrong type.
pub fn apply_attrs<T: Serialize + serde::de::DeserializeOwned>(base: &T, attrs: &StyleAttrs) -> Result<T, String> {
    if attrs.is_empty() {
        return serde_json::from_value(serde_json::to_value(base).map_err(|e| e.to_string())?).map_err(|e| e.to_string());
    }
    let mut v = serde_json::to_value(base).map_err(|e| e.to_string())?;
    let Value::Object(m) = &mut v else { return Err("style is not an object".into()) };
    for (k, x) in attrs {
        if k == REF_KEY {
            continue;
        }
        if !m.contains_key(k) {
            return Err(format!("unknown style attribute {k:?}"));
        }
        m.insert(k.clone(), x.clone());
    }
    serde_json::from_value(v).map_err(|e| e.to_string())
}

/// Attributes of `style` that differ from `base` (the style reference is ignored).
pub fn diff_attrs<T: Serialize>(style: &T, base: &T) -> StyleAttrs {
    let (a, b) = (object(style), object(base));
    a.into_iter().filter(|(k, v)| b.get(k) != Some(v)).collect()
}

/// Every attribute of a character style (a full set, for "redefine from text").
pub fn char_attrs(s: &CharStyle) -> StyleAttrs {
    object(s)
}

/// Every attribute of a paragraph style.
pub fn para_attrs(s: &ParagraphStyle) -> StyleAttrs {
    object(s)
}

/// Checks that `attrs` only names fields of `T` with values of the right type.
pub fn validate<T: Serialize + serde::de::DeserializeOwned + Default>(attrs: &StyleAttrs) -> Result<(), String> {
    apply_attrs(&T::default(), attrs).map(|_| ())
}

impl TextStyles {
    pub fn is_empty(&self) -> bool {
        self.character.is_empty() && self.paragraph.is_empty() && self.basic.char_attrs.is_empty() && self.basic.para_attrs.is_empty()
    }

    pub fn char_style(&self, id: u32) -> Option<&CharacterStyleDef> {
        self.character.iter().find(|s| s.id == id)
    }

    pub fn char_style_mut(&mut self, id: u32) -> Option<&mut CharacterStyleDef> {
        self.character.iter_mut().find(|s| s.id == id)
    }

    /// Paragraph style by id (0 = Basic Paragraph).
    pub fn para_style(&self, id: u32) -> Option<&ParagraphStyleDef> {
        if id == 0 { Some(&self.basic) } else { self.paragraph.iter().find(|s| s.id == id) }
    }

    pub fn para_style_mut(&mut self, id: u32) -> Option<&mut ParagraphStyleDef> {
        if id == 0 { Some(&mut self.basic) } else { self.paragraph.iter_mut().find(|s| s.id == id) }
    }

    /// A fresh character style id, or `None` when the stored ids exhaust the `u32` id space.
    pub fn next_char_id(&self) -> Option<u32> {
        self.character.iter().map(|s| s.id).max().unwrap_or(0).checked_add(1)
    }

    /// A fresh paragraph style id, or `None` when the stored ids exhaust the `u32` id space.
    pub fn next_para_id(&self) -> Option<u32> {
        self.paragraph.iter().map(|s| s.id).max().unwrap_or(0).checked_add(1)
    }

    /// "Character Style N" / "Paragraph Style N" with the first unused N.
    pub fn unique_name(&self, paragraph: bool, base: &str) -> String {
        let taken = |n: &str| {
            if paragraph {
                n == BASIC_PARAGRAPH || self.paragraph.iter().any(|s| s.name == n)
            } else {
                n == NO_CHARACTER_STYLE || self.character.iter().any(|s| s.name == n)
            }
        };
        (1..).map(|i| format!("{base} {i}")).find(|n| !taken(n)).unwrap_or_default()
    }

    /// Paragraph attributes from Basic Paragraph and `para` (a missing id falls back to Basic
    /// Paragraph). The result references `para`.
    pub fn resolve_para(&self, para: Option<u32>) -> ParagraphStyle {
        let mut s = apply_attrs(&ParagraphStyle::default(), &self.basic.para_attrs).unwrap_or_default();
        if let Some(def) = para.and_then(|id| self.para_style(id)) {
            s = apply_attrs(&s, &def.para_attrs).unwrap_or(s);
        }
        s.style_sheet = para.filter(|id| *id != 0 && self.para_style(*id).is_some());
        s
    }

    /// Character attributes a run gets from its styles alone: defaults (with `default_family`
    /// for an unset family) ← Basic Paragraph ← paragraph style ← character style. The result
    /// references `chr`.
    pub fn resolve_char(&self, para: Option<u32>, chr: Option<u32>, default_family: &str) -> CharStyle {
        let base = CharStyle { font_family: default_family.to_string(), ..Default::default() };
        let mut s = apply_attrs(&base, &self.basic.char_attrs).unwrap_or(base);
        if let Some(def) = para.and_then(|id| self.para_style(id)).filter(|d| d.id != 0) {
            s = apply_attrs(&s, &def.char_attrs).unwrap_or(s);
        }
        let chr = chr.filter(|id| self.char_style(*id).is_some());
        if let Some(def) = chr.and_then(|id| self.char_style(id)) {
            s = apply_attrs(&s, &def.attrs).unwrap_or(s);
        }
        s.style_sheet = chr;
        s
    }

    /// The local overrides of a run (attributes that differ from its styles).
    pub fn char_overrides(&self, run: &CharStyle, para: Option<u32>, default_family: &str) -> StyleAttrs {
        let base = self.resolve_char(para, run.style_sheet, default_family);
        diff_attrs(run, &base)
    }

    /// The local overrides of a paragraph.
    pub fn para_overrides(&self, p: &ParagraphStyle) -> StyleAttrs {
        diff_attrs(p, &self.resolve_para(p.style_sheet))
    }

    /// Re-resolves a run against (possibly changed) styles, keeping its overrides: `old` gives
    /// the styles as they were when the overrides were made.
    pub fn restyle_char(
        &self,
        old: &TextStyles,
        run: &CharStyle,
        old_para: Option<u32>,
        new_para: Option<u32>,
        new_chr: Option<u32>,
        family: &str,
    ) -> CharStyle {
        let overrides = old.char_overrides(run, old_para, family);
        let base = self.resolve_char(new_para, new_chr, family);
        apply_attrs(&base, &overrides).unwrap_or(base)
    }

    /// Re-resolves a paragraph against (possibly changed) styles, keeping its overrides.
    pub fn restyle_para(&self, old: &TextStyles, p: &ParagraphStyle, new_para: Option<u32>) -> ParagraphStyle {
        let overrides = old.para_overrides(p);
        let base = self.resolve_para(new_para);
        let mut s = apply_attrs(&base, &overrides).unwrap_or(base);
        s.style_sheet = new_para.filter(|id| *id != 0 && self.para_style(*id).is_some());
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text::TextAlign;
    use serde_json::json;

    fn attrs(v: Value) -> StyleAttrs {
        v.as_object().unwrap().clone()
    }

    fn styles() -> TextStyles {
        let mut s = TextStyles::default();
        s.character.push(CharacterStyleDef { id: 1, name: "Big".into(), attrs: attrs(json!({"size_pt": 40.0, "underline": true})) });
        s.paragraph.push(ParagraphStyleDef {
            id: 1,
            name: "Head".into(),
            para_attrs: attrs(json!({"align": "Center"})),
            char_attrs: attrs(json!({"size_pt": 20.0, "weight": 700})),
        });
        s
    }

    #[test]
    fn resolution_layers_styles() {
        let s = styles();
        let none = s.resolve_char(None, None, "Inter");
        assert_eq!((none.font_family.as_str(), none.size_pt, none.style_sheet), ("Inter", 12.0, None));
        let head = s.resolve_char(Some(1), None, "Inter");
        assert_eq!((head.size_pt, head.weight), (20.0, 700));
        let big = s.resolve_char(Some(1), Some(1), "Inter");
        assert_eq!((big.size_pt, big.weight, big.underline, big.style_sheet), (40.0, 700, true, Some(1)));
        // Missing ids fall back.
        assert_eq!(s.resolve_char(Some(9), Some(9), "Inter").style_sheet, None);
        let p = s.resolve_para(Some(1));
        assert_eq!((p.align, p.style_sheet), (TextAlign::Center, Some(1)));
        assert_eq!(s.resolve_para(None).align, TextAlign::Left);
    }

    #[test]
    fn overrides_survive_restyling() {
        let old = styles();
        let mut run = old.resolve_char(None, Some(1), "Inter");
        assert!(old.char_overrides(&run, None, "Inter").is_empty());
        run.italic = true;
        let ov = old.char_overrides(&run, None, "Inter");
        assert_eq!(ov, attrs(json!({"italic": true})));
        // Redefine "Big" to 50 pt: the run follows and keeps its italic override.
        let mut new = old.clone();
        new.char_style_mut(1).unwrap().attrs.insert("size_pt".into(), json!(50.0));
        let r = new.restyle_char(&old, &run, None, None, Some(1), "Inter");
        assert_eq!((r.size_pt, r.italic, r.underline), (50.0, true, true));
        // Clearing the character style keeps overrides on the paragraph basis.
        let r = new.restyle_char(&old, &run, None, None, None, "Inter");
        assert_eq!((r.size_pt, r.italic, r.underline, r.style_sheet), (12.0, true, false, None));
    }

    #[test]
    fn attrs_validate_and_diff() {
        assert!(validate::<CharStyle>(&attrs(json!({"size_pt": 3.0}))).is_ok());
        assert!(validate::<CharStyle>(&attrs(json!({"size_pt": "big"}))).is_err());
        assert!(validate::<CharStyle>(&attrs(json!({"nope": 1}))).is_err());
        let a = CharStyle { size_pt: 30.0, style_sheet: Some(3), ..Default::default() };
        assert_eq!(diff_attrs(&a, &CharStyle::default()), attrs(json!({"size_pt": 30.0})));
        assert!(!char_attrs(&a).contains_key("style_sheet"));
    }

    #[test]
    fn names_and_ids() {
        let s = styles();
        assert_eq!(s.unique_name(false, "Character Style"), "Character Style 1");
        assert_eq!((s.next_char_id(), s.next_para_id()), (Some(2), Some(2)));
        assert_eq!(s.para_style(0).unwrap().name, BASIC_PARAGRAPH);
        let v = serde_json::to_value(&s).unwrap();
        let back: TextStyles = serde_json::from_value(v).unwrap();
        assert_eq!(back, s);
        let empty: TextStyles = serde_json::from_str("{}").unwrap();
        assert_eq!(empty.basic.name, BASIC_PARAGRAPH);
    }

    #[test]
    fn exhausted_style_ids_survive_serialization() {
        let mut s = TextStyles::default();
        s.character.push(CharacterStyleDef { id: u32::MAX, ..Default::default() });
        s.paragraph.push(ParagraphStyleDef { id: u32::MAX, ..Default::default() });
        let value = serde_json::to_value(&s).unwrap();
        let loaded: TextStyles = serde_json::from_value(value).unwrap();
        assert_eq!(loaded.next_char_id(), None);
        assert_eq!(loaded.next_para_id(), None);
    }
}
