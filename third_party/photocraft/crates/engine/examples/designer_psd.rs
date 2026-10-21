//! Generate a realistic "designer" PSD for Layers-panel QA (#143, #144): about 150 layers in
//! groups nested four deep, long and short names (Latin, accented, CJK), many layers with
//! effects, clipping masks, layer and vector masks, locks, links, smart objects, adjustment,
//! fill, shape and type layers, hidden layers and a mix of open and closed groups. It is built
//! through engine commands and saved through the real PSD exporter.
//!
//! ```sh
//! cargo run -p photocraft-engine --example designer_psd -- out.psd
//! ```

use photocraft_engine::Session;
use serde_json::{Value, json};

const NAMES: &[&str] = &[
    "BG",
    "a",
    "Button / Primary / Hover state with a very long descriptive name that keeps going",
    "背景のテクスチャ 夕焼け",
    "Überschrift – Größe ½ (Entwurf)",
    "Shadow copy 3",
    "Rectangle 12",
    "Photo — Model portrait, retouched final FINAL v7 (approved by client)",
    "图层 12 副本",
    "Icon",
    "Highlight",
    "Card / Product thumbnail / Hover",
    "Ελληνικά κείμενο",
    "Glow",
    "Divider line",
    "카드 배경",
];

struct Gen {
    s: Session,
    n: usize,
}

impl Gen {
    fn run(&mut self, id: &str, p: Value) -> Value {
        match self.s.execute(id, p.clone()) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("{id} {p}: {e}");
                Value::Null
            }
        }
    }

    fn active(&self) -> u64 {
        self.s.active().and_then(|d| d.active_layer).map_or(0, |l| l.0)
    }

    /// One leaf layer of a kind picked from the running counter; returns its id.
    fn leaf(&mut self) -> u64 {
        let i = self.n;
        self.n += 1;
        let name = format!("{} {}", NAMES[i % NAMES.len()], if i.is_multiple_of(3) { String::new() } else { format!("#{i}") }).trim().to_string();
        let (x, y) = ((i * 97 % 1400) as i64, (i * 53 % 1000) as i64);
        let colour = format!("#{:02x}{:02x}{:02x}", (i * 47 % 256), (i * 91 % 256), (i * 23 % 256));
        let id = match i % 17 {
            3 => {
                let v = self.run("type.create", json!({"x": x, "y": y + 40, "text": format!("Headline {i}: Autumn collection"), "size": 36, "name": name}));
                v.get("layer").and_then(Value::as_u64).unwrap_or_else(|| self.active())
            }
            7 => {
                self.run("layer.newAdjustmentLayer.curves", json!({"points": [[0, 0], [128, 150], [255, 255]]}));
                let id = self.active();
                self.run("layer.setProps", json!({"layer": id, "name": format!("Curves {i}")}));
                id
            }
            11 => {
                self.run("shape.create", json!({"kind": "roundedRect", "rect": [x, y, 220, 90], "fill": colour, "name": name}));
                self.active()
            }
            14 => {
                self.run("layer.newFillLayer.solidColor", json!({"color": colour}));
                self.active()
            }
            _ => {
                self.run("layer.new.layer", json!({"name": name}));
                self.run("select.rect", json!({"x": x, "y": y, "width": 180 + i % 5 * 40, "height": 120 + i % 3 * 30}));
                self.run("edit.fill", json!({"contents": "color", "color": colour}));
                self.run("select.deselect", json!({}));
                self.active()
            }
        };
        if i % 4 == 1 {
            self.run("layer.layerStyle.dropShadow", json!({"layer": id, "distance": 6, "size": 10}));
            if i % 8 == 1 {
                self.run("layer.select", json!({"layer": id}));
                self.run("layer.layerStyle.stroke", json!({"size": 2, "color": "#ffffff", "add": true}));
            }
        }
        if i % 6 == 2 {
            self.run("layer.layerMask.revealAll", json!({"layer": id}));
        }
        if i % 10 == 5 {
            self.run("layer.select", json!({"layer": id}));
            let path = json!({"subpaths": [{"knots": [[x, y], [x + 300, y], [x + 300, y + 200], [x, y + 200]]}]});
            self.run("layer.vectorMask.add", json!({"layer": id, "path": path}));
        }
        if i % 9 == 4 {
            self.run("layer.setProps", json!({"layer": id, "locks": {"all": true}}));
        } else if i % 9 == 6 {
            self.run("layer.setProps", json!({"layer": id, "locks": {"position": true}}));
        }
        if i % 8 == 3 {
            self.run("layer.setProps", json!({"layer": id, "visible": false}));
        }
        if i % 12 == 10 {
            self.run("layer.setProps", json!({"layer": id, "blend": "Multiply"}));
        }
        if i % 13 == 6 {
            self.run("layer.select", json!({"layer": id}));
            self.run("layer.smartObjects.convertToSmartObject", json!({"layer": id}));
        }
        self.s.active().and_then(|d| d.active_layer).map_or(id, |l| l.0)
    }

    /// `count` items above the active layer, the item at `nest` (if any) a group built one
    /// level deeper, then everything grouped as `name`. Returns the group's id.
    fn group(&mut self, name: &str, depth: usize, count: usize, open: bool) -> u64 {
        let mut ids = Vec::new();
        for k in 0..count {
            if depth < 3 && k == count / 2 {
                let sub = format!("{name} › Level {} {}", depth + 2, ["Cards", "サブグループ", "Details", "Variants"][depth % 4]);
                ids.push(self.group(&sub, depth + 1, count.saturating_sub(2).max(3), !self.n.is_multiple_of(3)));
            } else {
                let id = self.leaf();
                // Clip every 7th layer to the one below it (inside the same group).
                if self.n.is_multiple_of(7) && !ids.is_empty() {
                    self.run("layer.setProps", json!({"layer": id, "clipped": true}));
                }
                ids.push(id);
            }
        }
        self.run("layer.select", json!({"layer": ids[0]}));
        for id in &ids[1..] {
            self.run("layer.select", json!({"layer": id, "mode": "toggle"}));
        }
        let g = self.run("layer.groupLayers", json!({"name": name}))["layer"].as_u64().unwrap_or_else(|| self.active());
        if depth == 1 && self.n.is_multiple_of(2) {
            self.run("layer.layerStyle.dropShadow", json!({"layer": g}));
        }
        self.run("layer.setExpanded", json!({"layer": g, "expanded": open}));
        self.run("layer.select", json!({"layer": g}));
        g
    }
}

fn main() -> Result<(), String> {
    let out = std::env::args().nth(1).unwrap_or_else(|| "designer.psd".into());
    let mut g = Gen { s: Session::new(), n: 0 };
    g.run("file.new", json!({"width": 1600, "height": 1200, "name": "designer"}));
    let groups = [
        ("Footer", 6, false),
        ("Retouching — dodge & burn, frequency separation", 7, true),
        ("Typography / 本文", 6, false),
        ("Product Grid — Autumn/Winter 2026 campaign (final, do not edit)", 8, true),
        ("ヒーローセクション", 7, true),
        ("Navigation", 5, false),
        ("Header / Logo lockup", 6, true),
    ];
    for (name, count, open) in groups {
        g.group(name, 0, count, open);
    }
    for _ in 0..10 {
        g.leaf();
    }
    // Two linked layers at the top.
    let a = g.leaf();
    let b = g.leaf();
    g.run("layer.select", json!({"layer": a}));
    g.run("layer.select", json!({"layer": b, "mode": "toggle"}));
    g.run("layer.linkLayers", json!({}));
    let doc = g.s.active().map(|d| (*d.doc).clone()).ok_or("no document")?;
    let layers = doc.walk().len();
    let bytes = photocraft_io::export(&doc, &out, &Default::default()).map_err(|e| e.to_string())?.bytes;
    std::fs::write(&out, &bytes).map_err(|e| e.to_string())?;
    println!("wrote {out}: {layers} layers, {} bytes", bytes.len());
    Ok(())
}
