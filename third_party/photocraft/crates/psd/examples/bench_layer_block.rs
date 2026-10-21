//! Serialization of one 24 MP raw RGB layer in Lr16/Lr32, without compression or disk I/O.
//! Run with `cargo run --release -p photocraft-psd --example bench_layer_block`.
use photocraft_psd::*;
use std::hint::black_box;
use std::time::Instant;

fn main() -> Result<()> {
    for depth in [16, 32] {
        let mut file = testgen::small(Version::Psd, Compression::Raw);
        file.header = Header::new(Version::Psd, 6000, 4000, 3, depth, ColorMode::Rgb);
        let plane_bytes = 6000 * 4000 * usize::from(depth / 8);
        let mut layer = LayerRecord { rect: Rect::from_xywh(0, 0, 6000, 4000), name: b"RGB".to_vec(), ..Default::default() };
        for id in 0..3 {
            layer.channels.push(ChannelData { id, compression: Some(Compression::Raw), data: vec![0; plane_bytes] });
        }
        file.layer_info = Some(LayerInfo { merged_alpha: false, layers: vec![layer], padding: None });
        file.layer_info_placement =
            LayerInfoPlacement::GlobalBlock { index: 0, signature: *b"8BIM", key: if depth == 16 { *b"Lr16" } else { *b"Lr32" }, padding: None };
        file.image_data = ImageData { compression: Compression::Raw, data: vec![0; plane_bytes * 3] };
        black_box(file.to_bytes()?);
        let mut samples = Vec::new();
        let mut size = 0;
        for _ in 0..11 {
            let t = Instant::now();
            let bytes = file.to_bytes()?;
            samples.push(t.elapsed().as_secs_f64() * 1000.0);
            size = bytes.len();
            black_box(bytes);
        }
        samples.sort_by(f64::total_cmp);
        println!(
            "{depth}-bit, {} bytes: p50={:.3} ms p95={:.3} ms",
            size,
            samples.get(5).copied().unwrap_or_default(),
            samples.get(10).copied().unwrap_or_default()
        );
    }
    Ok(())
}
