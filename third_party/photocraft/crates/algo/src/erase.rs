//! Magic Eraser: erase a wand [`Region`] to transparency (or, on a layer with locked transparency,
//! to a colour), tile by tile.
//!
//! Works in the surface's own channels at any depth and in any colour model: only the alpha
//! channel changes (or, under a transparency lock, the colour channels move toward `lock_color`).

use photocraft_color::{read_sample, write_sample};
use photocraft_geom::Rect;
use photocraft_raster::Surface;

use crate::selection::Region;

/// Erases `region` from `target` at `amount` (0–1, the tool's opacity), weighted by the selection's
/// coverage. With `lock_color` (native colour channels, e.g. the background colour converted to the
/// surface's model) the pixels are painted toward that colour instead and alpha is kept, as
/// Photoshop does on layers with locked transparency. Returns the damaged rectangle.
pub fn magic_erase(target: &mut Surface, region: &Region, amount: f32, selection: Option<&Surface>, lock_color: Option<&[f32]>) -> Rect {
    let amount = if amount.is_finite() { amount.clamp(0.0, 1.0) } else { 0.0 };
    let bbox = region.bbox;
    let bw = bbox.width() as usize;
    let Some(pixels) = bw.checked_mul(bbox.height() as usize) else { return Rect::EMPTY };
    if bbox.is_empty() || amount <= 0.0 || region.mask.len() != pixels {
        return Rect::EMPTY;
    }
    let fmt = target.format();
    let nc = fmt.mode.color_channels();
    let alpha = fmt.alpha.then_some(nc);
    // Without alpha and without a lock colour there is nothing an eraser can do.
    if alpha.is_none() && lock_color.is_none() {
        return Rect::EMPTY;
    }
    let lock: Option<Vec<f32>> = lock_color.map(|c| (0..nc).map(|i| c.get(i).copied().unwrap_or(0.0)).collect());
    let rects: Vec<Rect> = bbox.tiles().map(|tc| tc.rect().intersect(&bbox)).filter(|r| !r.is_empty()).collect();
    let src: &Surface = target;
    let (sample, bpp) = (fmt.sample, fmt.bytes_per_pixel());
    // Each tile is edited as encoded bytes (only the touched samples are decoded), in parallel;
    // the results are copied back serially.
    let work = |r: Rect| -> Option<(Rect, Vec<u8>)> {
        let w = r.width() as usize;
        let cov = |x: i32, y: i32| region.mask.get((y - bbox.y0) as usize * bw + (x - bbox.x0) as usize).copied().unwrap_or(0);
        // Skip tiles the region doesn't touch.
        if !(r.y0..r.y1).any(|y| (r.x0..r.x1).any(|x| cov(x, y) != 0)) {
            return None;
        }
        let mut bytes = src.to_interleaved(r);
        let sel = selection.map(|s| (s.channels(), s.read_region(r)));
        for y in r.y0..r.y1 {
            for x in r.x0..r.x1 {
                let c = cov(x, y);
                if c == 0 {
                    continue;
                }
                let i = (y - r.y0) as usize * w + (x - r.x0) as usize;
                let s = sel.as_ref().map_or(1.0, |(sc, v)| v.get(i * sc).copied().unwrap_or(0.0));
                let k = f32::from(c) / 255.0 * amount * s;
                if k <= 0.0 {
                    continue;
                }
                let Some(px) = bytes.get_mut(i * bpp..(i + 1) * bpp) else { continue };
                match (&lock, alpha) {
                    (Some(col), _) => {
                        for (ch, t) in col.iter().enumerate() {
                            let v = read_sample(px, sample, ch);
                            write_sample(px, sample, ch, v + (t - v) * k);
                        }
                    }
                    (None, Some(a)) => {
                        let v = read_sample(px, sample, a) * (1.0 - k);
                        if k >= 1.0 || v <= 0.0 {
                            // Fully erased pixels become the empty pixel, so tiles can be pruned.
                            px.fill(0);
                        } else {
                            write_sample(px, sample, a, v);
                        }
                    }
                    (None, None) => {}
                }
            }
        }
        Some((r, bytes))
    };
    #[cfg(not(target_arch = "wasm32"))]
    let done: Vec<(Rect, Vec<u8>)> = {
        use rayon::prelude::*;
        rects.into_par_iter().filter_map(work).collect()
    };
    #[cfg(target_arch = "wasm32")]
    let done: Vec<(Rect, Vec<u8>)> = rects.into_iter().filter_map(work).collect();
    let mut dmg = Rect::EMPTY;
    for (r, bytes) in done {
        target.write_interleaved(r, &bytes);
        dmg = dmg.union(&r);
    }
    if lock.is_none() {
        target.prune();
    }
    dmg
}

#[cfg(test)]
mod tests {
    use super::*;
    use photocraft_color::{ColorMode, PixelFormat, SampleType};
    use photocraft_raster::from_rgba;

    fn region(bbox: Rect, v: u8) -> Region {
        Region { bbox, mask: vec![v; bbox.width() as usize * bbox.height() as usize] }
    }

    #[test]
    fn erases_alpha_in_every_format() {
        for mode in [ColorMode::Rgb, ColorMode::Cmyk, ColorMode::Grayscale, ColorMode::Lab] {
            for sample in [SampleType::U8, SampleType::U16, SampleType::F32] {
                let fmt = PixelFormat::new(mode, sample, true);
                let mut s = Surface::new(fmt);
                s.fill_rect(Rect::new(0, 0, 600, 300), &from_rgba(&fmt, [0.2, 0.5, 0.7, 1.0]));
                let before = s.pixel(10, 10);
                let d = magic_erase(&mut s, &region(Rect::new(0, 0, 300, 300), 255), 1.0, None, None);
                assert_eq!(d, Rect::new(0, 0, 300, 300));
                assert_eq!(s.rgba(10, 10)[3], 0.0, "{fmt:?}");
                assert_eq!(s.pixel(400, 10), before, "outside the region is untouched {fmt:?}");
                // Half opacity halves alpha and keeps colour.
                let mut h = Surface::new(fmt);
                h.fill_rect(Rect::new(0, 0, 10, 10), &from_rgba(&fmt, [0.2, 0.5, 0.7, 1.0]));
                magic_erase(&mut h, &region(Rect::new(0, 0, 10, 10), 255), 0.5, None, None);
                let p = h.pixel(5, 5);
                let a = fmt.mode.color_channels();
                assert!((p[a] - 0.5).abs() < 0.01, "{fmt:?} {p:?}");
                assert!((p[0] - before[0]).abs() < 1e-6, "colour kept {fmt:?}");
            }
        }
    }

    #[test]
    fn respects_selection_and_lock() {
        let fmt = PixelFormat::RGBA8;
        let mut s = Surface::new(fmt);
        s.fill_rect(Rect::new(0, 0, 20, 20), &[1.0, 0.0, 0.0, 1.0]);
        let mut sel = Surface::new(PixelFormat::GRAY8);
        sel.fill_rect(Rect::new(0, 0, 10, 20), &[1.0]);
        magic_erase(&mut s, &region(Rect::new(0, 0, 20, 20), 255), 1.0, Some(&sel), None);
        assert_eq!(s.rgba(5, 5)[3], 0.0);
        assert_eq!(s.rgba(15, 5)[3], 1.0);
        // Locked transparency paints the lock colour, alpha kept.
        let mut l = Surface::new(fmt);
        l.fill_rect(Rect::new(0, 0, 20, 20), &[1.0, 0.0, 0.0, 1.0]);
        magic_erase(&mut l, &region(Rect::new(0, 0, 20, 20), 255), 1.0, None, Some(&[1.0, 1.0, 1.0]));
        assert_eq!(l.rgba(5, 5), [1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn oversized_region_mask_is_rejected_without_overflow() {
        let mut s = Surface::new(PixelFormat::RGBA8);
        s.fill_rect(Rect::new(0, 0, 1, 1), &[1.0, 0.0, 0.0, 1.0]);
        let before = s.pixel(0, 0);
        let oversized = Region { bbox: Rect::new(i32::MIN, i32::MIN, i32::MAX, i32::MAX), mask: vec![255] };
        assert_eq!(magic_erase(&mut s, &oversized, 1.0, None, None), Rect::EMPTY);
        assert_eq!(s.pixel(0, 0), before);
    }

    #[test]
    fn degenerate_inputs_do_nothing() {
        let mut s = Surface::new(PixelFormat::RGBA8);
        s.fill_rect(Rect::new(0, 0, 4, 4), &[1.0, 0.0, 0.0, 1.0]);
        let orig = s.clone();
        // Mask length mismatch, zero/NaN amount, empty box.
        assert!(magic_erase(&mut s, &Region { bbox: Rect::new(0, 0, 4, 4), mask: vec![255; 3] }, 1.0, None, None).is_empty());
        assert!(magic_erase(&mut s, &region(Rect::new(0, 0, 4, 4), 255), f32::NAN, None, None).is_empty());
        assert!(magic_erase(&mut s, &region(Rect::EMPTY, 255), 1.0, None, None).is_empty());
        assert_eq!(s, orig);
        // No alpha channel and no lock colour: nothing to erase.
        let mut g = Surface::new(PixelFormat::GRAY8);
        assert!(magic_erase(&mut g, &region(Rect::new(0, 0, 4, 4), 255), 1.0, None, None).is_empty());
    }
}
