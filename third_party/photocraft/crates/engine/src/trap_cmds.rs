//! Image › Trap. Trapping is a CMYK prepress step (see `photocraft_algo::trap`). Photoshop requires
//! a flattened CMYK image; we trap the active pixel layer's inks over the canvas. One history step.

use photocraft_color::ColorMode;
use photocraft_doc::LayerContent;
use serde_json::{Value, json};

use crate::commands::CommandSpec;
use crate::{EngineError, Result, Session};

fn enabled(s: &Session) -> std::result::Result<(), String> {
    match s.active() {
        Some(d) if d.doc.mode == ColorMode::Cmyk => Ok(()),
        Some(_) => Err("Trap needs a CMYK document (Image › Mode › CMYK Color)".into()),
        None => Err("no document".into()),
    }
}

fn trap(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "image.trap";
    let width = p.get("width").and_then(Value::as_f64).unwrap_or(1.0).round().max(0.0) as usize;
    if s.active().ok_or(EngineError::NoDocument)?.doc.mode != ColorMode::Cmyk {
        return Err(EngineError::BadParams { cmd: cmd.into(), msg: "Trap needs a CMYK document".into() });
    }
    s.edit("Trap", |doc, active| {
        let id = active.ok_or(EngineError::NoDocument)?;
        let bounds = doc.bounds();
        let l = doc.layer_mut(id).ok_or(EngineError::NoLayer(id))?;
        let LayerContent::Raster(surf) = &mut l.content else {
            return Err(EngineError::BadParams {
                cmd: cmd.into(),
                msg: format!("Trap needs a pixel layer (active layer is {} {})", l.content.article(), l.content.kind_name()),
            });
        };
        if bounds.is_empty() {
            return Ok(());
        }
        let ch = surf.channels();
        let (w, h) = (bounds.width() as usize, bounds.height() as usize);
        let mut data = surf.read_region(bounds);
        photocraft_algo::trap::trap(&mut data, w, h, ch, width);
        surf.write_region(bounds, &data);
        Ok(())
    })?;
    Ok(json!({"trapped": true, "width": width}))
}

pub fn specs() -> Vec<CommandSpec> {
    vec![CommandSpec {
        id: "image.trap",
        label: "Trap…",
        menu: &["Image"],
        shortcut: None,
        params: r#"{width:px>=1=1} → {trapped,width}: spread inks at colour edges (CMYK only)"#,
        enabled,
        journal: true,
        run: |s, p| trap(s, p),
    }]
}

#[cfg(test)]
mod tests {
    use super::*;
    use photocraft_geom::Rect;

    #[test]
    fn traps_a_cmyk_document() {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 8, "height": 4, "mode": "cmyk"})).unwrap();
        // Paint cyan | magenta halves on the Background.
        s.edit("paint", |doc, active| {
            let id = active.unwrap();
            let fmt = doc.pixel_format();
            let l = doc.layer_mut(id).unwrap();
            if let LayerContent::Raster(surf) = &mut l.content {
                let ch = fmt.channels();
                let mut data = vec![0.0f32; 8 * 4 * ch];
                for y in 0..4 {
                    for x in 0..8 {
                        let px = &mut data[(y * 8 + x) * ch..][..ch];
                        if x < 4 {
                            px[0] = 1.0;
                        } else {
                            px[1] = 1.0;
                        }
                        if fmt.alpha {
                            px[ch - 1] = 1.0;
                        }
                    }
                }
                surf.write_region(Rect::new(0, 0, 8, 4), &data);
            }
            Ok(())
        })
        .unwrap();

        let r = s.execute("image.trap", json!({"width": 1})).unwrap();
        assert_eq!(r["trapped"], true);
        // The seam now has overlapping inks.
        let st = s.active().unwrap();
        let id = st.active_layer.unwrap();
        if let LayerContent::Raster(surf) = &st.doc.layer(id).unwrap().content {
            let ch = surf.channels();
            let d = surf.read_region(Rect::new(0, 0, 8, 4));
            assert!(d[3 * ch + 1] > 0.5, "magenta trapped into the cyan edge");
            assert!(d[4 * ch] > 0.5, "cyan trapped into the magenta edge");
        }
        // One history step.
        assert_eq!(st.history.entries().last().map(|e| e.as_str()), Some("Trap"));
    }

    #[test]
    fn rejects_non_cmyk() {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 8, "height": 4, "mode": "rgb"})).unwrap();
        assert!(s.execute("image.trap", json!({"width": 1})).is_err());
    }
}
