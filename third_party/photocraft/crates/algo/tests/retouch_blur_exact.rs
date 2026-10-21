//! Frozen scalar convolution from main 5dac5c6, used as an exact-output oracle.
use photocraft_algo::retouch::local_blur;

#[test]
fn matches_scalar_blur_bit_for_bit_across_layouts_edges_and_hdr() {
    for (w, h) in [(1, 1), (3, 7), (37, 29)] {
        for ch in 1..=5 {
            for alpha in std::iter::once(None).chain((0..ch).map(Some)) {
                let data: Vec<f32> = (0..w * h * ch)
                    .map(|i| if Some(i % ch) == alpha { [0.0, 1e-8, 0.25, 0.75, 1.0][(i / ch) % 5] } else { ((i * 73 + 19) % 113) as f32 / 17.0 - 2.0 })
                    .collect();
                for win in [(0, 0, w, h), (w / 2, h / 2, w, h), (0, 0, 1, 1), (0, 0, 0, h)] {
                    for sigma in [0.0, 0.25, 1.0, 3.0, 4.0, 12.0] {
                        let expected = scalar_local_blur(&data, w, h, ch, alpha, win, sigma);
                        let actual = local_blur(&data, w, h, ch, alpha, win, sigma);
                        assert_eq!(actual.len(), expected.len());
                        for (i, (a, b)) in actual.iter().zip(&expected).enumerate() {
                            assert_eq!(a.to_bits(), b.to_bits(), "{w}x{h} ch={ch} alpha={alpha:?} win={win:?} sigma={sigma} sample={i}");
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn empty_channel_layout_remains_empty() {
    assert!(local_blur(&[], 5, 5, 0, None, (0, 0, 5, 5), 4.0).is_empty());
}

#[allow(clippy::too_many_arguments)]
fn scalar_local_blur(data: &[f32], w: usize, h: usize, ch: usize, alpha: Option<usize>, win: (usize, usize, usize, usize), sigma: f32) -> Vec<f32> {
    let (x0, y0, x1, y1) = win;
    let (ow, oh) = (x1 - x0, y1 - y0);
    let r = (sigma * 3.0).ceil().max(1.0) as i32;
    let k: Vec<f32> = (-r..=r).map(|i| (-((i * i) as f32) / (2.0 * sigma * sigma).max(1e-6)).exp()).collect();
    let ks: f32 = k.iter().sum();
    let k: Vec<f32> = k.into_iter().map(|v| v / ks).collect();
    let get = |x: i32, y: i32, out: &mut [f32]| {
        let (xx, yy) = (x.clamp(0, w as i32 - 1) as usize, y.clamp(0, h as i32 - 1) as usize);
        let i = (yy * w + xx) * ch;
        out.copy_from_slice(&data[i..i + ch]);
        if let Some(a) = alpha {
            let av = out[a];
            for (c, v) in out.iter_mut().enumerate() {
                if c != a {
                    *v *= av;
                }
            }
        }
    };
    // Horizontal pass over the window rows extended vertically by r.
    let th = oh + 2 * r as usize;
    let mut tmp = vec![0.0f32; ow * th * ch];
    let mut px = vec![0.0f32; ch];
    for ty in 0..th {
        let y = y0 as i32 + ty as i32 - r;
        for ox in 0..ow {
            let o = (ty * ow + ox) * ch;
            for (i, kv) in k.iter().enumerate() {
                get(x0 as i32 + ox as i32 + i as i32 - r, y, &mut px);
                for c in 0..ch {
                    tmp[o + c] += px[c] * kv;
                }
            }
        }
    }
    let mut out = vec![0.0f32; ow * oh * ch];
    for oy in 0..oh {
        for ox in 0..ow {
            let o = (oy * ow + ox) * ch;
            for (i, kv) in k.iter().enumerate() {
                let s = ((oy + i) * ow + ox) * ch;
                for c in 0..ch {
                    out[o + c] += tmp[s + c] * kv;
                }
            }
            if let Some(a) = alpha {
                let av = out[o + a];
                for c in 0..ch {
                    if c != a {
                        out[o + c] = if av > 1e-6 { out[o + c] / av } else { data[((y0 + oy) * w + x0 + ox) * ch + c] };
                    }
                }
            }
        }
    }
    out
}
