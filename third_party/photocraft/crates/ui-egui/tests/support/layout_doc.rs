//! A synthetic "designer's layout" document (#125, #128): a web/landing-page style PSD with
//! nested groups (pass-through and isolated), dozens of text layers in several fonts and sizes,
//! shape layers with strokes, smart objects, layer effects (drop shadow, stroke, gradient overlay),
//! adjustment layers and layer masks. Shared by `examples/layout_bench.rs` and the
//! `layout_perf` tests (`#[path]`-included; deterministic, no files).
//!
//! [`Spec::full`] is a 4000×3000 document with 174 layers (60 of them type); the tests use a small
//! one (`cards` sets how many product cards, the bulk of the layer count).

#![allow(dead_code)]

use photocraft_color::{BlendMode, Color, ColorMode, SampleType};
use photocraft_doc::{Document, Effect, FxCommon, FxPaint, Gradient, Layer, LayerContent, LayerId, LayerMask, PixelFormat, StrokeFx, StrokePosition};
use photocraft_engine::Session;
use photocraft_geom::{Rect, Size};
use serde_json::{Value, json};

/// Size and content knobs.
#[derive(Clone, Copy, Debug)]
pub struct Spec {
    pub width: u32,
    pub height: u32,
    /// Product cards (7 layers each: group, card, image smart object, title, body, price, badge).
    pub cards: usize,
    /// Gallery tiles (masked pixel layers).
    pub tiles: usize,
}

impl Spec {
    /// The benchmark document: 4000×3000, 174 layers, 60 text layers.
    pub fn full() -> Self {
        Spec { width: 4000, height: 3000, cards: 16, tiles: 24 }
    }
    /// A quick one for tests: same structure, fewer and smaller layers.
    pub fn small() -> Self {
        Spec { width: 800, height: 600, cards: 3, tiles: 4 }
    }
}

/// Handles to representative layers (for driving the UI).
#[derive(Clone, Debug)]
pub struct Handles {
    pub pixel: LayerId,
    pub text: LayerId,
    pub shape: LayerId,
    pub group: LayerId,
    pub smart: LayerId,
    /// A closed group (collapsed in the Layers panel).
    pub closed_group: LayerId,
}

fn exec(s: &mut Session, id: &str, p: Value) -> Value {
    s.execute(id, p).unwrap_or_else(|e| panic!("{id}: {e}"))
}

fn id_of(v: &Value) -> LayerId {
    LayerId(v.get("layer").or_else(|| v.get("id")).and_then(Value::as_u64).expect("layer id in result"))
}

fn active(s: &Session) -> LayerId {
    s.active().and_then(|st| st.active_layer).expect("active layer")
}

/// Deterministic pseudo-random in [0, 1).
fn rnd(k: u64) -> f32 {
    let mut x = k.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5_4A32_D192_ED03;
    x ^= x >> 31;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 29;
    (x >> 40) as f32 / (1u64 << 24) as f32
}

fn hex(c: [f32; 3]) -> String {
    format!("#{:02x}{:02x}{:02x}", (c[0] * 255.0) as u8, (c[1] * 255.0) as u8, (c[2] * 255.0) as u8)
}

fn drop_shadow(distance: f32, size: f32) -> Effect {
    let Effect::DropShadow(mut ds) = Effect::default_drop_shadow() else { unreachable!("default drop shadow") };
    ds.distance = distance;
    ds.size = size;
    ds.common.opacity = 0.35;
    Effect::DropShadow(ds)
}

fn stroke_fx(size: f32, c: [f32; 3]) -> Effect {
    Effect::Stroke(StrokeFx {
        common: FxCommon::new(BlendMode::Normal, 1.0),
        size,
        position: StrokePosition::Outside,
        paint: FxPaint::Color(Color::rgb(c[0], c[1], c[2])),
    })
}

fn gradient_overlay(a: [f32; 3], b: [f32; 3]) -> Effect {
    Effect::GradientOverlay {
        common: FxCommon::new(BlendMode::Normal, 0.6),
        gradient: Gradient { stops: vec![(0.0, Color::rgb(a[0], a[1], a[2])), (1.0, Color::rgb(b[0], b[1], b[2]))], angle: 90.0, ..Default::default() },
        dither: false,
    }
}

/// A photo-ish pixel block (gradient + soft noise), `r` in document coordinates.
fn photo_layer(name: &str, fmt: PixelFormat, r: Rect, seed: u64) -> Layer {
    let mut l = Layer::raster(name, fmt);
    let s = l.surface_mut().expect("raster");
    let step = 16;
    let base = [rnd(seed), rnd(seed + 1), rnd(seed + 2)];
    let mut y = r.y0;
    while y < r.y1 {
        let mut x = r.x0;
        while x < r.x1 {
            let fx = (x - r.x0) as f32 / r.width().max(1) as f32;
            let fy = (y - r.y0) as f32 / r.height().max(1) as f32;
            let n = rnd(seed ^ ((x as u64) << 20) ^ y as u64) * 0.08;
            let px = [(base[0] * 0.7 + 0.3 * fx + n).min(1.0), (base[1] * 0.6 + 0.4 * fy + n).min(1.0), (base[2] * 0.8 + 0.2 * (1.0 - fx) + n).min(1.0), 1.0];
            s.fill_rect(Rect::new(x, y, (x + step).min(r.x1), (y + step).min(r.y1)), &px);
            x += step;
        }
        y += step;
    }
    l
}

fn take(doc: &mut Document, id: LayerId) -> Layer {
    doc.remove(id).unwrap_or_else(|| panic!("layer {id:?} missing"))
}

/// Builds the layout document (and the session that built it, for its text engine state).
pub fn build(spec: Spec) -> (Document, Handles) {
    let (w, h) = (spec.width as i32, spec.height as i32);
    let u = w as f32 / 4000.0; // layout unit: 1 at 4000 px wide
    let px = |v: f32| (v * u).round() as i32;
    let pt = |v: f32| (v * u).max(4.0);
    let doc = Document::with_background("Layout", Size::new(spec.width, spec.height), ColorMode::Rgb, SampleType::U8, Color::rgb(0.97, 0.97, 0.96));
    let fmt = doc.pixel_format();
    let mut s = Session::new();
    s.add_document(doc, None);
    let fonts = ["Inter", "JetBrains Mono", "Helvetica", "Georgia", "Arial"];
    let text = |s: &mut Session, x: i32, y: i32, t: &str, font: &str, size: f32, color: [f32; 3], extra: Value| -> LayerId {
        let mut p = json!({"x": x, "y": y, "text": t, "font": font, "size": size, "color": hex(color)});
        if let (Some(o), Some(e)) = (p.as_object_mut(), extra.as_object()) {
            o.extend(e.clone());
        }
        id_of(&exec(s, "type.create", p))
    };
    let shape = |s: &mut Session, kind: &str, r: [i32; 4], fill: [f32; 3], stroke: Option<(f32, [f32; 3])>, name: &str| -> LayerId {
        let mut p = json!({"kind": kind, "rect": r, "fill": hex(fill), "name": name, "radii": px(24.0)});
        if let Some((wd, c)) = stroke {
            p["stroke"] = json!({"width": wd, "color": hex(c), "align": "inside"});
        }
        exec(s, "shape.create", p);
        active(s)
    };
    let smart = |s: &mut Session, name: &str, r: Rect, seed: u64| -> LayerId {
        let st = s.active_mut().expect("doc");
        let mut doc = (*st.doc).clone();
        let l = photo_layer(name, fmt, r, seed);
        let id = doc.insert_above(None, l);
        st.doc = std::sync::Arc::new(doc);
        exec(s, "layer.select", json!({"layer": id.0}));
        id_of(&exec(s, "layer.smartObjects.convertToSmartObject", json!({"layer": id.0})))
    };
    let ink = [0.12, 0.13, 0.16];
    let accent = [0.85, 0.32, 0.18];

    // ---- header -------------------------------------------------------------------------------
    let logo = smart(&mut s, "Logo", Rect::new(px(120.0), px(60.0), px(420.0), px(220.0)), 1);
    let nav: Vec<LayerId> = ["Home", "Products", "Pricing", "Journal", "About", "Contact"]
        .iter()
        .enumerate()
        .map(|(i, t)| text(&mut s, px(1700.0 + i as f32 * 340.0), px(160.0), t, fonts[i % 2], pt(28.0), ink, json!({})))
        .collect();
    let rule = shape(&mut s, "rect", [0, px(260.0), w, px(4.0).max(1)], [0.85, 0.85, 0.85], None, "Rule");

    // ---- hero ---------------------------------------------------------------------------------
    let hero_img = photo_layer("Hero photo", fmt, Rect::new(0, px(270.0), w, px(1500.0)), 7);
    let hero_img_id = {
        let st = s.active_mut().expect("doc");
        let mut doc = (*st.doc).clone();
        let mut l = hero_img;
        let mut m = LayerMask::reveal_all();
        m.surface.fill_rect(Rect::new(0, px(1300.0), w, px(1500.0)), &[0.4]);
        l.mask = Some(m);
        l.effects.items = vec![gradient_overlay([0.1, 0.1, 0.3], [0.9, 0.5, 0.2])];
        let id = doc.insert_above(None, l);
        st.doc = std::sync::Arc::new(doc);
        id
    };
    let headline = text(&mut s, px(240.0), px(800.0), "Make things that last", fonts[3], pt(160.0), [1.0, 1.0, 1.0], json!({}));
    let subhead = text(
        &mut s,
        0,
        0,
        "Hand-finished goods for everyday use. Designed in small batches, built to be repaired, and shipped carbon-neutral.",
        fonts[0],
        pt(40.0),
        [0.95, 0.95, 0.95],
        json!({"box": [px(240.0), px(900.0), px(1800.0), px(260.0)]}),
    );
    let cta_bg = shape(&mut s, "roundedRect", [px(240.0), px(1200.0), px(520.0), px(130.0)], accent, Some((pt(6.0), [1.0, 1.0, 1.0])), "Button");
    let cta_txt = text(&mut s, px(300.0), px(1285.0), "Shop the collection", fonts[0], pt(36.0), [1.0, 1.0, 1.0], json!({"weight": 700}));
    let hue = id_of(&exec(&mut s, "layer.newAdjustmentLayer.hueSaturation", json!({"saturation": -15})));

    // ---- cards --------------------------------------------------------------------------------
    let cols = 4usize;
    let (cw, ch) = (px(860.0), px(1150.0));
    let mut cards: Vec<Vec<LayerId>> = Vec::new();
    for i in 0..spec.cards {
        let (cx, cy) = (px(160.0) + (i % cols) as i32 * px(940.0), px(1600.0) + (i / cols) as i32 * (ch / 4));
        let bg = shape(&mut s, "roundedRect", [cx, cy, cw, ch / 5], [1.0, 1.0, 1.0], Some((pt(3.0), [0.8, 0.8, 0.82])), "Card");
        let img = smart(&mut s, &format!("Product {}", i + 1), Rect::new(cx + px(20.0), cy + px(20.0), cx + cw - px(20.0), cy + ch / 9), 100 + i as u64);
        let title = text(&mut s, cx + px(30.0), cy + ch / 9 + px(50.0), &format!("Product {} — Oak", i + 1), fonts[i % fonts.len()], pt(34.0), ink, json!({}));
        let body = text(
            &mut s,
            0,
            0,
            "Solid oak, oiled by hand. Lifetime repairs.",
            fonts[(i + 1) % fonts.len()],
            pt(20.0),
            [0.4, 0.4, 0.42],
            json!({"box": [cx + px(30.0), cy + ch / 9 + px(70.0), cw - px(60.0), px(60.0)]}),
        );
        let price = text(&mut s, cx + cw - px(200.0), cy + ch / 9 + px(50.0), &format!("${}", 40 + i * 15), fonts[1], pt(30.0), accent, json!({}));
        let badge = shape(&mut s, "star", [cx + cw - px(120.0), cy + px(40.0), px(80.0), px(80.0)], accent, Some((pt(2.0), [1.0, 1.0, 1.0])), "Badge");
        cards.push(vec![bg, img, title, body, price, badge]);
    }

    // ---- gallery ------------------------------------------------------------------------------
    let mut tiles = Vec::new();
    for i in 0..spec.tiles {
        let (tx, ty) = (px(160.0) + (i % 8) as i32 * px(470.0), px(2350.0) + (i / 8) as i32 * px(200.0));
        tiles.push(photo_layer(&format!("Tile {}", i + 1), fmt, Rect::new(tx, ty, tx + px(440.0), ty + px(180.0)), 500 + i as u64));
    }
    let gallery_title = text(&mut s, px(160.0), px(2300.0), "From the workshop", fonts[3], pt(64.0), ink, json!({}));
    let curves = id_of(&exec(&mut s, "layer.newAdjustmentLayer.curves", json!({"points": [[0, 0], [100, 120], [255, 255]]})));

    // ---- footer -------------------------------------------------------------------------------
    let foot_bg = shape(&mut s, "rect", [0, px(2800.0), w, px(200.0)], [0.12, 0.13, 0.16], None, "Footer bg");
    let mut foot = Vec::new();
    for i in 0..2 {
        foot.push(text(
            &mut s,
            px(160.0) + (i % 5) as i32 * px(700.0),
            px(2870.0) + (i / 5) as i32 * px(60.0),
            ["Shipping", "Returns", "Repairs", "Wholesale", "Press", "Careers", "Privacy", "Terms", "Instagram", "Newsletter"][i],
            fonts[i % 3],
            pt(22.0),
            [0.85, 0.85, 0.86],
            json!({}),
        ));
    }
    let foot_icons: Vec<LayerId> =
        (0..5).map(|i| shape(&mut s, "ellipse", [w - px(600.0) + i * px(100.0), px(2870.0), px(60.0), px(60.0)], [0.9, 0.9, 0.9], None, "Icon")).collect();
    let levels = id_of(&exec(&mut s, "layer.newAdjustmentLayer.levels", json!({"inBlack": 4, "inWhite": 250, "gamma": 1.05})));

    // ---- assemble the tree (children bottom-first) --------------------------------------------
    let mut doc = (*s.active().expect("doc").doc).clone();
    let mut t = |id: LayerId| take(&mut doc, id);
    let mut header_kids = vec![t(rule), t(logo)];
    let nav_layers: Vec<Layer> = nav.iter().map(|&i| t(i)).collect();
    let mut nav_group = Layer::group("Navigation", nav_layers);
    nav_group.blend = BlendMode::Normal; // isolated
    header_kids.push(nav_group);
    let header = Layer::group("Header", header_kids);

    let mut headline_l = t(headline);
    headline_l.effects.items = vec![drop_shadow(px(12.0) as f32, px(24.0) as f32)];
    let mut cta_l = t(cta_bg);
    cta_l.effects.items = vec![drop_shadow(px(6.0) as f32, px(12.0) as f32)];
    let mut hue_l = t(hue);
    hue_l.clipped = true;
    let button = Layer::group("CTA", vec![cta_l, t(cta_txt)]);
    let hero_photo = t(hero_img_id);
    let mut hero = Layer::group("Hero", vec![hero_photo, hue_l, headline_l, t(subhead), button]);
    if let LayerContent::Group(g) = &mut hero.content {
        g.expanded = false;
    }
    let hero_id = hero.id;

    let mut card_groups = Vec::new();
    let z = LayerId(0);
    let mut handles = Handles { pixel: z, text: z, shape: z, group: z, smart: z, closed_group: z };
    for (i, c) in cards.iter().enumerate() {
        let mut bg = t(c[0]);
        bg.effects.items = vec![drop_shadow(px(8.0) as f32, px(18.0) as f32)];
        let mut img = t(c[1]);
        // Image masked to the card's top (layer mask).
        let b = photocraft_compose::layer_bounds(&img, Rect::new(0, 0, w, h));
        let mut m = LayerMask::reveal_all();
        m.surface.fill_rect(Rect::new(b.x0, b.y1 - (b.height() as i32 / 6), b.x1, b.y1), &[0.0]);
        img.mask = Some(m);
        let mut title = t(c[2]);
        if i % 3 == 0 {
            title.effects.items = vec![stroke_fx(2.0, [1.0, 1.0, 1.0])];
        }
        let mut badge = t(c[5]);
        badge.effects.items = vec![gradient_overlay([1.0, 0.8, 0.2], [0.9, 0.3, 0.1])];
        if i == 0 {
            handles.smart = img.id;
            handles.text = title.id;
            handles.shape = badge.id;
        }
        let g = Layer::group(format!("Card {}", i + 1), vec![bg, img, title, t(c[3]), t(c[4]), badge]);
        if i == 0 {
            handles.group = g.id;
        }
        card_groups.push(g);
    }
    let mut rows = Vec::new();
    for (r, chunk) in card_groups.chunks(cols).enumerate() {
        let mut row = Layer::group(format!("Row {}", r + 1), chunk.to_vec());
        row.blend = if r.is_multiple_of(2) { BlendMode::PassThrough } else { BlendMode::Normal };
        rows.push(row);
    }
    let products = Layer::group("Products", rows);

    let mut tile_layers = Vec::new();
    for (i, mut l) in tiles.into_iter().enumerate() {
        let b = photocraft_compose::layer_bounds(&l, Rect::new(0, 0, w, h));
        let mut m = LayerMask::reveal_all();
        m.surface.fill_rect(Rect::new(b.x0, b.y0, b.x0 + b.width() as i32 / 8, b.y1), &[0.0]);
        l.mask = Some(m);
        if i == 0 {
            handles.pixel = l.id;
        }
        tile_layers.push(l);
    }
    tile_layers.push(t(gallery_title));
    let mut curves_l = t(curves);
    curves_l.opacity = 0.8;
    tile_layers.push(curves_l);
    let gallery = Layer::group("Gallery", tile_layers);

    let mut footer_kids = vec![t(foot_bg)];
    footer_kids.extend(foot.iter().map(|&i| t(i)));
    let icons: Vec<Layer> = foot_icons.iter().map(|&i| t(i)).collect();
    footer_kids.push(Layer::group("Social", icons));
    let mut footer = Layer::group("Footer", footer_kids);
    if let LayerContent::Group(g) = &mut footer.content {
        g.expanded = false;
    }
    let lv = t(levels);
    let background = doc.layers.remove(0);
    assert!(doc.layers.is_empty(), "unassembled layers: {:?}", doc.layers.iter().map(|l| &l.name).collect::<Vec<_>>());
    doc.layers = vec![background, gallery, products, hero, header, footer, lv];
    handles.closed_group = hero_id;
    (doc, handles)
}

/// Every layer in the tree.
pub fn count(doc: &Document) -> (usize, usize) {
    let all = doc.walk();
    let texts = all.iter().filter(|(_, _, l)| matches!(l.content, LayerContent::Text(_))).count();
    (all.len(), texts)
}
