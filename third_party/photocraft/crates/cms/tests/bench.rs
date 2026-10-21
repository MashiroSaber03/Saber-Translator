//! Throughput check: `cargo test -p photocraft-cms --release --test bench -- --ignored --nocapture`.

use photocraft_cms::{Builtin, Intent, Transform};

#[test]
#[ignore]
fn bench_srgb8_to_cmyk8() {
    let t0 = std::time::Instant::now();
    let t = Transform::new(Builtin::Srgb.profile(), Builtin::CoatedCmyk.profile(), Intent::RelativeColorimetric, true).unwrap();
    eprintln!("build: {:?}", t0.elapsed());
    let n = 6016 * 6016;
    let mut src = vec![0u8; n * 4];
    let mut x = 1u32;
    for v in src.iter_mut() {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        *v = x as u8;
    }
    let mut dst = vec![0u8; n * 5];
    let t1 = std::time::Instant::now();
    t.convert_u8(&src, 4, &mut dst, 5, true);
    eprintln!("convert 6016² RGBA8 → CMYKA8 (one buffer): {:?}", t1.elapsed());
    let t2 = std::time::Instant::now();
    for (s, d) in src.chunks(65536 * 4).zip(dst.chunks_mut(65536 * 5)) {
        t.convert_u8(s, 4, d, 5, true);
    }
    eprintln!("convert per 256² tile: {:?}", t2.elapsed());
}
