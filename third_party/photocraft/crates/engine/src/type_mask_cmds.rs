//! `type.createSelection`: the Horizontal and Vertical Type Mask tools' result. Typed text
//! becomes a selection in the shape of its glyphs (their anti-aliased coverage) instead of a
//! type layer, as Photoshop's Type Mask tools do.
//!
//! The tools edit a temporary type layer in one coalesced history step and commit with
//! `{"layer": id, "removeLayer": true}`, so the whole session is one undo step that leaves no
//! layer behind. Scripts can skip the layer and pass `type.create`'s keys directly.

use photocraft_algo::selection::{self as sel, SelectionMode};
use photocraft_doc::text::AntiAlias;
use photocraft_doc::{Document, LayerContent, LayerId, TextLayer};
use photocraft_geom::Rect;
use serde_json::{Value, json};

use crate::commands::CommandSpec;
use crate::{EngineError, Result, Session};

const CMD: &str = "type.createSelection";

fn bad(msg: impl Into<String>) -> EngineError {
    EngineError::BadParams { cmd: CMD.into(), msg: msg.into() }
}

fn has_doc(s: &Session) -> std::result::Result<(), String> {
    s.active().map(|_| ()).ok_or_else(|| "no document open".into())
}

fn antialias(p: &Value) -> Result<Option<AntiAlias>> {
    Ok(match p.get("antialias") {
        None => None,
        Some(v) => Some(match v.as_str() {
            Some("none") => AntiAlias::None,
            Some("sharp") => AntiAlias::Sharp,
            Some("crisp") => AntiAlias::Crisp,
            Some("strong") => AntiAlias::Strong,
            Some("smooth") => AntiAlias::Smooth,
            _ => return Err(bad("antialias must be none, sharp, crisp, strong or smooth")),
        }),
    })
}

/// Glyph coverage of `t` (0..=1 per pixel) over its rendered bounds inside the canvas. The text
/// colour's own alpha doesn't thin the selection: every run renders opaque.
fn coverage(doc: &Document, t: &TextLayer) -> (Vec<f32>, Rect) {
    let mut t = t.clone();
    for r in &mut t.runs {
        r.style.color.alpha = 1.0;
    }
    let mut eng = photocraft_text::shared().lock().unwrap_or_else(|e| e.into_inner());
    let surface = eng.render(&t, doc.resolution_dpi, doc.pixel_format()).1.surface;
    let area = surface.content_bounds().intersect(&doc.bounds());
    if area.is_empty() {
        return (Vec::new(), Rect::EMPTY);
    }
    let mut px = vec![[0.0f32; 4]; area.width() as usize * area.height() as usize];
    surface.read_rgba_into(area, &mut px);
    (px.iter().map(|p| p[3]).collect(), area)
}

/// Coverage of `area`'s mask spread over `to` (which contains `area`), zero elsewhere.
fn widen(m: &[f32], area: Rect, to: Rect) -> Vec<f32> {
    if area == to {
        return m.to_vec();
    }
    let (w, aw) = (to.width() as usize, area.width() as usize);
    let mut out = vec![0.0; w * to.height() as usize];
    for (y, row) in m.chunks_exact(aw.max(1)).enumerate() {
        let start = (area.y0 - to.y0) as usize * w + y * w + (area.x0 - to.x0) as usize;
        if let Some(dst) = out.get_mut(start..start + aw) {
            dst.copy_from_slice(row);
        }
    }
    out
}

fn create_selection(s: &mut Session, p: &Value) -> Result<Value> {
    let mode = match p.get("operation").map(|v| v.as_str()) {
        None | Some(Some("new" | "replace")) => SelectionMode::Replace,
        Some(Some("add")) => SelectionMode::Add,
        Some(Some("subtract")) => SelectionMode::Subtract,
        Some(Some("intersect")) => SelectionMode::Intersect,
        _ => return Err(bad("operation must be new, add, subtract or intersect")),
    };
    let aa = antialias(p)?;
    let remove = match p.get("removeLayer") {
        None => false,
        Some(v) => v.as_bool().ok_or_else(|| bad("removeLayer must be true or false"))?,
    };
    let doc = s.active().ok_or(EngineError::NoDocument)?.doc.clone();
    let layer = match p.get("layer") {
        None | Some(Value::Null) => None,
        Some(v) => Some(LayerId(v.as_u64().ok_or_else(|| bad("layer must be a layer id"))?)),
    };
    if remove && layer.is_none() {
        return Err(bad("removeLayer needs a layer"));
    }
    let mut t = match layer {
        Some(id) => match &doc.layer(id).ok_or(EngineError::NoLayer(id))?.content {
            LayerContent::Text(t) => t.clone(),
            _ => return Err(bad(format!("layer {} is not a type layer", id.0))),
        },
        None => crate::type_cmds::new_text_layer(s, p, CMD)?,
    };
    if let Some(aa) = aa {
        t.antialias = aa;
    }
    let (mask, text_area) = coverage(&doc, &t);
    let selected = s.edit("Type Mask", |doc, active| {
        let old = doc.selection.as_ref().map(|o| o.content_bounds().intersect(&doc.bounds())).unwrap_or(Rect::EMPTY);
        // Replace and Intersect leave nothing outside the text; Add and Subtract keep the rest.
        let area = match mode {
            SelectionMode::Add | SelectionMode::Subtract if !old.is_empty() => {
                if text_area.is_empty() {
                    old
                } else {
                    old.union(&text_area)
                }
            }
            _ => text_area,
        };
        doc.selection = if area.is_empty() { None } else { sel::combine(doc.selection.as_ref(), &widen(&mask, text_area, area), area, mode) };
        if let (true, Some(id)) = (remove, layer) {
            let neighbours = crate::layer_multi_cmds::deletion_neighbours(doc, id);
            doc.remove(id).ok_or(EngineError::NoLayer(id))?;
            if doc.layers.is_empty() {
                return Err(EngineError::Other("a document must keep at least one layer".into()));
            }
            if active.is_some_and(|current| doc.layer(current).is_none()) {
                *active = neighbours.into_iter().find(|candidate| doc.layer(*candidate).is_some());
            }
        }
        Ok(doc.selection.as_ref().map(|m| m.content_bounds()))
    })?;
    // The Type Mask tools' whole typing session is this one step.
    if let Some(st) = s.active_mut() {
        st.history.set_current_label("Type Mask");
    }
    Ok(json!({ "selected": selected.is_some(), "bounds": selected.map(|r| [r.x0, r.y0, r.x1, r.y1]) }))
}

pub fn specs() -> Vec<CommandSpec> {
    vec![CommandSpec {
        id: CMD,
        label: "Type Mask",
        menu: &[],
        shortcut: None,
        params: r##"{"layer":id? (a type layer whose glyphs become the selection),"removeLayer":bool=false (delete that layer in the same step: the Type Mask tools' temporary text),"operation":"new|add|subtract|intersect"="new","antialias":"none|sharp|crisp|strong|smooth"?, …without "layer", type.create's keys: "text","x","y" or "box","orientation","font","size","align",…}"##,
        enabled: has_doc,
        journal: true,
        run: create_selection,
    }]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn session() -> Session {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 300, "height": 300})).unwrap();
        s
    }

    fn bounds(s: &Session) -> Option<Rect> {
        s.active().unwrap().doc.selection.as_ref().map(|m| m.content_bounds())
    }

    #[test]
    fn typed_text_selects_its_glyphs_without_a_layer() {
        let mut s = session();
        let layers = s.active().unwrap().doc.layers.len();
        let text = json!({"text": "I", "x": 40, "y": 120, "size": 48});
        s.execute(CMD, text.clone()).unwrap();
        let sel = bounds(&s).expect("the glyph is selected");
        assert_eq!(s.active().unwrap().doc.layers.len(), layers, "no layer added");
        // The same text as a type layer covers the same pixels.
        s.execute("edit.undo", json!({})).unwrap();
        assert!(bounds(&s).is_none(), "one undo step");
        let r = s.execute("type.create", text).unwrap();
        let b: Vec<i32> = r["bounds"].as_array().unwrap().iter().map(|v| v.as_i64().unwrap() as i32).collect();
        assert_eq!([sel.x0, sel.y0, sel.x1, sel.y1], [b[0], b[1], b[2], b[3]]);

        // Vertical: the word runs down the column, so its selection is taller than wide.
        let mut s = session();
        s.execute(CMD, json!({"text": "HELLO", "x": 150, "y": 40, "size": 48, "orientation": "vertical"})).unwrap();
        let v = bounds(&s).unwrap();
        assert!(v.height() > 2 * v.width(), "{v:?}");
    }

    #[test]
    fn bad_params_are_errors() {
        let mut s = session();
        for p in [json!({"layer": "x"}), json!({"removeLayer": true}), json!({"operation": 3}), json!({"antialias": "blurry"}), json!({"layer": 999})] {
            assert!(s.execute(CMD, p.clone()).is_err(), "{p}");
        }
        assert!(Session::new().execute(CMD, json!({"text": "I"})).is_err());
    }
}
