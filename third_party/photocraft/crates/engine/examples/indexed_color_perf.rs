//! Local Indexed Color timing probe. Never commits or uploads the input image.
//! Run in release: `cargo run -p photocraft-engine --release --example indexed_color_perf -- image.png`.
//! Reports palette construction, diffusion and the complete CPU preview path (no GPU upload).

use photocraft_algo::quantize::{self, Dither, Forced, PaletteKind};
use photocraft_engine::Session;
use serde_json::json;
use std::{error::Error, time::Instant};

fn elapsed(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

fn pixel_hash(px: &[[f32; 4]]) -> String {
    let mut hash = blake3::Hasher::new();
    for p in px {
        for value in p {
            hash.update(&value.to_bits().to_le_bytes());
        }
    }
    hash.finalize().to_hex().to_string()
}

fn main() -> Result<(), Box<dyn Error>> {
    let path = std::env::args().nth(1).ok_or("provide a local image path")?;
    let bytes = photocraft_format::read::read_file(std::path::Path::new(&path))?;
    let doc = photocraft_io::import(&path, &bytes)?.document;
    let px = photocraft_compose::flatten(&doc).px;
    println!("{}", json!({"width":doc.size.width,"height":doc.size.height,"preview_factor":1}));
    for count in [8usize, 16, 32, 64, 128, 256] {
        for run in 0..4 {
            let start = Instant::now();
            let clear = px.iter().any(|p| p[3] < 0.5);
            let mut pal = quantize::build_palette(&px, PaletteKind::Selective, if clear { count.saturating_sub(1).max(2) } else { count }, Forced::BlackWhite)?;
            let transparent = if clear && pal.len() < 256 {
                pal.push([255; 3]);
                Some(pal.len() - 1)
            } else {
                None
            };
            let palette_ms = elapsed(start);
            let mut out = px.clone();
            let start = Instant::now();
            let indices = quantize::quantize(&mut out, doc.size.width as usize, &pal, Dither::Diffusion, 0.75, transparent);
            let diffusion_ms = elapsed(start);
            std::hint::black_box(indices);
            let start = Instant::now();
            let proxy = photocraft_compose::proxy::proxy_document(&doc, 1);
            let mut session = Session::new();
            session.add_document(proxy, None);
            session.execute("image.mode.indexedColor", json!({"colors":count}))?;
            let result = session.active().ok_or("preview has no document")?;
            let composed = photocraft_compose::flatten(&result.doc);
            let preview_ms = elapsed(start);
            println!(
                "{}",
                json!({"colors":count,"run":run,"warmup":run==0,"palette_ms":palette_ms,
                "diffusion_ms":diffusion_ms,"preview_ms":preview_ms,"palette":pal,"pixels_hash":pixel_hash(&out),
                "preview_hash":pixel_hash(&composed.px)})
            );
        }
    }
    Ok(())
}
