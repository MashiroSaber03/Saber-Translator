//! BEFORE/AFTER helper for the banded interleave/deinterleave: `--serial` runs the old
//! sequential implementation inline, the default runs the current parallel one.
fn main() {
    use photocraft_color::SampleType;
    let serial = std::env::args().any(|a| a == "--serial");
    for (label, s, ch) in [("u16 rgb", SampleType::U16, 3usize), ("f32 rgba", SampleType::F32, 4)] {
        let bps = s.bytes();
        let n = 20_000_000usize;
        let planes: Vec<Vec<u8>> = (0..ch).map(|c| (0..n * bps).map(|i| (i * 7 + c) as u8).collect()).collect();
        let refs: Vec<Option<&[u8]>> = planes.iter().map(|p| Some(&p[..])).collect();
        let fill: Vec<Vec<u8>> = (0..ch).map(|_| vec![0u8; bps]).collect();
        let invert = [false; 4];
        let t0 = std::time::Instant::now();
        let interleaved = if serial {
            let mut out = vec![0u8; n * ch * bps];
            for (c, plane) in refs.iter().enumerate() {
                for i in 0..n {
                    let dst = &mut out[(i * ch + c) * bps..(i * ch + c + 1) * bps];
                    if let Some(p) = plane
                        && p.len() >= (i + 1) * bps
                    {
                        let src = &p[i * bps..(i + 1) * bps];
                        for k in 0..bps {
                            dst[k] = src[bps - 1 - k];
                        }
                    }
                }
            }
            out
        } else {
            photocraft_io::pixels::interleave(&refs, &fill, n, s, &invert)
        };
        let t1 = t0.elapsed().as_secs_f64() * 1000.0;
        let t2 = std::time::Instant::now();
        if serial {
            let mut acc: Vec<Vec<u8>> = Vec::new();
            for _ in 0..ch {
                acc.push(Vec::with_capacity(n * bps));
            }
            let mut tmp = [0u8; 4];
            for i in 0..n {
                for (c, plane) in acc.iter_mut().enumerate() {
                    let src = &interleaved[(i * ch + c) * bps..(i * ch + c + 1) * bps];
                    tmp[..bps].copy_from_slice(src);
                    plane.extend(tmp[..bps].iter().rev());
                }
            }
        } else {
            let _ = photocraft_io::pixels::deinterleave(&interleaved, ch, s, &invert);
        }
        let t3 = t2.elapsed().as_secs_f64() * 1000.0;
        println!("{label}: interleave {t1:.1} ms, deinterleave {t3:.1} ms (20 MP, release, {})", if serial { "serial (before)" } else { "parallel (after)" });
    }
}
