//! Timing for the computational-photography algorithms on a ~24 MP (6000×4000) image:
//! Camera Raw develop, Lens Correction, Adaptive Wide Angle, HDR merge + tone mapping, Crop and
//! Straighten detection, and a 4-image panorama registration + blend (2 MP sources).
//! `cargo run --release -p photocraft-algo --example bench_photo [case-substring]`
use std::time::Instant;

use photocraft_algo::camera_raw::{CameraRaw, Wheel, develop};
use photocraft_algo::hdr::{self, MergeOptions, ToneMethod};
use photocraft_algo::lens::{self, LensCorrection};
use photocraft_algo::panorama::{self, AlignOptions, Layout, RoiImage};
use photocraft_algo::tone::HdrToning;
use photocraft_algo::transform::Interp;
use photocraft_algo::wideangle::{self, Constraint, Orientation, WideAngle, WideModel};
use photocraft_color::PixelFormat;
use photocraft_geom::Rect;
use photocraft_raster::Surface;

fn pattern(w: usize, h: usize) -> Vec<[f32; 4]> {
    (0..w * h)
        .map(|i| {
            let (x, y) = (i % w, i / w);
            let t = ((x * 7 + y * 3) % 251) as f32 / 251.0;
            [(x % 256) as f32 / 255.0 * 0.6 + 0.2 * t, ((x * 3 + y) / 11 % 256) as f32 / 255.0, ((x ^ y) % 256) as f32 / 255.0, 1.0]
        })
        .collect()
}

fn time<T>(name: &str, only: &Option<String>, f: impl FnOnce() -> T) -> Option<T> {
    if only.as_ref().is_some_and(|o| !name.contains(o.as_str())) {
        return None;
    }
    let t0 = Instant::now();
    let r = f();
    println!("{name:<40} {:>9.1} ms", t0.elapsed().as_secs_f64() * 1000.0);
    Some(r)
}

fn main() {
    let only = std::env::args().nth(1);
    let (w, h) = (6000usize, 4000usize);
    let px = pattern(w, h);
    let r = Rect::new(0, 0, w as i32, h as i32);
    let mut s = Surface::new(PixelFormat::RGBA8);
    let bytes: Vec<u8> = px.iter().flat_map(|q| q.map(|v| (v * 255.0) as u8)).collect();
    s.write_interleaved(r, &bytes);

    time("camera raw: basic (wb, exposure, tone)", &only, || {
        let mut a = px.clone();
        develop(
            &mut a,
            w,
            h,
            &CameraRaw {
                temperature: 20.0,
                exposure: 0.3,
                contrast: 20.0,
                highlights: -40.0,
                shadows: 40.0,
                whites: 10.0,
                blacks: -10.0,
                vibrance: 20.0,
                ..Default::default()
            },
            false,
        );
    });
    time("camera raw: clarity + texture + dehaze", &only, || {
        let mut a = px.clone();
        develop(&mut a, w, h, &CameraRaw { clarity: 30.0, texture: 20.0, dehaze: 25.0, ..Default::default() }, false);
    });
    time("camera raw: hsl + grading + curve", &only, || {
        let mut a = px.clone();
        let mut hs = [0.0; 8];
        hs[2] = 30.0;
        develop(
            &mut a,
            w,
            h,
            &CameraRaw {
                hsl_sat: hs,
                grade_shadows: Wheel { hue: 220.0, sat: 30.0, lum: 0.0 },
                point_curve: vec![[0.0, 10.0], [128.0, 140.0], [255.0, 250.0]],
                ..Default::default()
            },
            false,
        );
    });
    time("camera raw: sharpen + NR + grain + vignette", &only, || {
        let mut a = px.clone();
        develop(
            &mut a,
            w,
            h,
            &CameraRaw { sharpen_amount: 40.0, noise_luminance: 30.0, noise_color: 25.0, grain_amount: 20.0, vignette_amount: -30.0, ..Default::default() },
            false,
        );
    });
    time("lens correction (distortion, CA, vignette, persp)", &only, || {
        lens::correct(
            &s,
            r,
            &LensCorrection {
                profile: Some(lens::generic_profile(24.0)),
                distortion: 10.0,
                red_cyan: 20.0,
                vignette_amount: 30.0,
                vertical: -20.0,
                angle: 1.5,
                ..Default::default()
            },
        )
    });
    time("adaptive wide angle (fisheye, 2 constraints)", &only, || {
        let p = WideAngle {
            model: WideModel::Fisheye,
            focal_length: 12.0,
            constraints: vec![
                Constraint { a: [500.0, 600.0], b: [5500.0, 600.0], orientation: Orientation::Horizontal },
                Constraint { a: [700.0, 500.0], b: [700.0, 3500.0], orientation: Orientation::Vertical },
            ],
            ..Default::default()
        };
        wideangle::apply(&s, r, &p, Interp::Bicubic)
    });
    time("hdr: merge 3 exposures + local adaptation", &only, || {
        let shots: Vec<Vec<[f32; 4]>> =
            [0.5f32, 1.0, 2.0].iter().map(|k| px.iter().map(|q| [(q[0] * k).min(1.0), (q[1] * k).min(1.0), (q[2] * k).min(1.0), 1.0]).collect()).collect();
        let refs: Vec<&[[f32; 4]]> = shots.iter().map(Vec::as_slice).collect();
        let t0 = Instant::now();
        let mut m = hdr::merge(w, h, &refs, &MergeOptions { exposures: vec![0.5, 1.0, 2.0], remove_ghosts: true, ghost_base: None, response: None });
        println!("  merge {:.1} ms", t0.elapsed().as_secs_f64() * 1000.0);
        hdr::tone_map(&mut m.px, w, h, &ToneMethod::LocalAdaptation(HdrToning { radius: 7.0, strength: 0.52, ..Default::default() }));
    });
    time("crop and straighten: detection", &only, || photocraft_algo::scancrop::find_photos(w, h, &px));
    time("panorama: 4 × 2 MP register + blend", &only, || {
        let (sw, sh) = (1700usize, 1200usize);
        let views: Vec<Vec<f32>> = (0..4)
            .map(|k| {
                (0..sw * sh)
                    .map(|i| {
                        let q = px[(i / sw + 300) * w + i % sw + 1100 * k];
                        0.3 * q[0] + 0.6 * q[1] + 0.1 * q[2]
                    })
                    .collect()
            })
            .collect();
        let prepared: Vec<panorama::PreparedImage> = views.iter().map(|v| (sw, sh, v.clone(), None)).collect();
        let t0 = Instant::now();
        let feats = panorama::detect_many(&prepared, 800);
        let matches = panorama::match_pairs(&feats);
        let al = panorama::align(&feats, &matches, &AlignOptions { layout: Layout::Reposition, focal: None, reference: None, geometric: false });
        println!(
            "  register {:.1} ms ({} pairs, placed {})",
            t0.elapsed().as_secs_f64() * 1000.0,
            matches.len(),
            al.as_ref().map_or(0, |a| a.placements.iter().flatten().count())
        );
        let rois: Vec<RoiImage> = views
            .iter()
            .enumerate()
            .map(|(k, v)| RoiImage { x0: 1100 * k as i32, y0: 0, w: sw, h: sh, ch: 1, px: v.clone(), alpha: vec![1.0; sw * sh] })
            .collect();
        let cw = 1100 * 3 + sw;
        let wts: Vec<Vec<f32>> = (0..4i32)
            .map(|k| {
                (0..sw * sh)
                    .map(|i| {
                        let x = (i % sw) as i32 + 1100 * k;
                        if (x / 1100).min(3) == k { 1.0 } else { 0.0 }
                    })
                    .collect()
            })
            .collect();
        let t1 = Instant::now();
        let _ = panorama::multiband(cw, sh, &rois, &wts, panorama::blend_levels(cw, sh));
        println!("  multiband {:.1} ms", t1.elapsed().as_secs_f64() * 1000.0);
    });
}
