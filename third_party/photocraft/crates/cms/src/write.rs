//! ICC v4.3 profile writer (ICC.1:2010), used for built-in and synthetic profiles.

use crate::clut::Clut;
use crate::curve::Curve;
use crate::math;
use crate::pipeline::Stage;
use crate::profile::{LutKind, Pcs, Profile};

fn s15(v: f64) -> [u8; 4] {
    ((v * 65536.0).round() as i32).to_be_bytes()
}

fn xyz_type(v: [f64; 3]) -> Vec<u8> {
    let mut o = b"XYZ \0\0\0\0".to_vec();
    for c in v {
        o.extend_from_slice(&s15(c));
    }
    o
}

fn mluc(text: &str) -> Vec<u8> {
    let units: Vec<u16> = text.encode_utf16().collect();
    let mut o = b"mluc\0\0\0\0".to_vec();
    o.extend_from_slice(&1u32.to_be_bytes());
    o.extend_from_slice(&12u32.to_be_bytes());
    o.extend_from_slice(b"enUS");
    o.extend_from_slice(&((units.len() * 2) as u32).to_be_bytes());
    o.extend_from_slice(&28u32.to_be_bytes());
    for u in units {
        o.extend_from_slice(&u.to_be_bytes());
    }
    o
}

fn u16q(v: f32) -> u16 {
    (v.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16
}

/// `curv` or `para` element for a curve.
fn curve_elem(c: &Curve) -> Vec<u8> {
    match c {
        Curve::Identity => {
            let mut o = b"curv\0\0\0\0".to_vec();
            o.extend_from_slice(&0u32.to_be_bytes());
            o
        }
        Curve::Gamma(g) => {
            let mut o = b"para\0\0\0\0".to_vec();
            o.extend_from_slice(&[0, 0, 0, 0]);
            o.extend_from_slice(&s15(*g));
            o
        }
        Curve::Parametric { kind, p } => {
            let mut o = b"para\0\0\0\0".to_vec();
            o.extend_from_slice(&kind.to_be_bytes());
            o.extend_from_slice(&[0, 0]);
            for v in p.iter().take(Curve::param_count(*kind)) {
                o.extend_from_slice(&s15(*v));
            }
            o
        }
        Curve::Table(t) => {
            let mut o = b"curv\0\0\0\0".to_vec();
            o.extend_from_slice(&(t.len() as u32).to_be_bytes());
            for v in t {
                o.extend_from_slice(&u16q(*v).to_be_bytes());
            }
            o
        }
    }
}

fn pad4(v: &mut Vec<u8>) {
    while !v.len().is_multiple_of(4) {
        v.push(0);
    }
}

fn curve_table(c: &Curve, n: usize) -> Vec<f32> {
    match c {
        Curve::Table(t) if t.len() == n => t.clone(),
        _ => c.sample(n),
    }
}

fn table_len(cs: &[Curve]) -> usize {
    cs.iter()
        .map(|c| match c {
            Curve::Table(t) => t.len(),
            Curve::Identity => 2,
            _ => 1024,
        })
        .max()
        .unwrap_or(2)
        .clamp(2, 4096)
}

/// Splits a LUT's stages into `(matrix, input curves, clut, output curves)` for `mft1`/`mft2`.
type LutParts<'a> = (Option<Vec<f64>>, &'a [Curve], &'a Clut, &'a [Curve]);

fn lut_parts(stages: &[Stage]) -> Option<LutParts<'_>> {
    let mut i = 0;
    let mut mat = None;
    if let Some(Stage::Matrix { rows: 3, cols: 3, m, .. }) = stages.first() {
        mat = Some(m.clone());
        i = 1;
    }
    match &stages[i..] {
        [Stage::Curves(a), Stage::Clut(c), Stage::Curves(b)] => Some((mat, a, c, b)),
        _ => None,
    }
}

/// `None` unless the stages are `[matrix?] curves, clut, curves` with a uniform grid.
fn encode_mft(stages: &[Stage], wide: bool) -> Option<Vec<u8>> {
    let (mat, a, clut, b) = lut_parts(stages)?;
    let g = *clut.grid.first()?;
    if !clut.grid.iter().all(|x| *x == g) {
        return None;
    }
    let mut o = if wide { b"mft2\0\0\0\0".to_vec() } else { b"mft1\0\0\0\0".to_vec() };
    o.extend_from_slice(&[clut.inputs as u8, clut.outputs as u8, g as u8, 0]);
    let m = mat.unwrap_or_else(|| vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    for v in &m {
        o.extend_from_slice(&s15(*v));
    }
    let (n_in, n_out) = if wide { (table_len(a), table_len(b)) } else { (256, 256) };
    if wide {
        o.extend_from_slice(&(n_in as u16).to_be_bytes());
        o.extend_from_slice(&(n_out as u16).to_be_bytes());
    }
    let put = |o: &mut Vec<u8>, v: f32| {
        if wide {
            o.extend_from_slice(&u16q(v).to_be_bytes());
        } else {
            o.push((v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
        }
    };
    for c in a {
        for v in curve_table(c, n_in) {
            put(&mut o, v);
        }
    }
    for v in &clut.data {
        put(&mut o, *v);
    }
    for c in b {
        for v in curve_table(c, n_out) {
            put(&mut o, v);
        }
    }
    Some(o)
}

/// `None` when the stages aren't a layout `lutAtoB`/`lutBtoA` can store in this direction.
fn encode_ab(stages: &[Stage], a2b: bool, inputs: usize, outputs: usize) -> Option<Vec<u8>> {
    let mut a_curves: Option<&[Curve]> = None;
    let mut clut: Option<&Clut> = None;
    let mut m_curves: Option<&[Curve]> = None;
    let mut matrix: Option<(&Vec<f64>, &Vec<f64>)> = None;
    let mut b_curves: Option<&[Curve]> = None;
    let mut it = stages.iter().peekable();
    if a2b {
        // [A, CLUT] [M, matrix] B
        if let (Some(Stage::Curves(a)), Some(Stage::Clut(_))) = (stages.first(), stages.get(1)) {
            a_curves = Some(a);
            it.next();
            if let Some(Stage::Clut(c)) = it.next() {
                clut = Some(c);
            }
        }
        let rest: Vec<&Stage> = it.collect();
        match rest.as_slice() {
            [Stage::Curves(m), Stage::Matrix { m: mm, offset, .. }, Stage::Curves(b)] => {
                m_curves = Some(m.as_slice());
                matrix = Some((mm, offset));
                b_curves = Some(b.as_slice());
            }
            [Stage::Curves(b)] => b_curves = Some(b.as_slice()),
            _ => return None,
        }
    } else {
        // B [matrix, M] [CLUT, A]
        let v: Vec<&Stage> = it.collect();
        let mut k = 0;
        if let Some(Stage::Curves(b)) = v.first() {
            b_curves = Some(b.as_slice());
            k = 1;
        }
        if let (Some(Stage::Matrix { m: mm, offset, .. }), Some(Stage::Curves(m))) = (v.get(k), v.get(k + 1)) {
            matrix = Some((mm, offset));
            m_curves = Some(m.as_slice());
            k += 2;
        }
        if let (Some(Stage::Clut(c)), Some(Stage::Curves(a))) = (v.get(k), v.get(k + 1)) {
            clut = Some(c);
            a_curves = Some(a.as_slice());
            k += 2;
        }
        if k != v.len() {
            return None;
        }
    }
    let mut o = if a2b { b"mAB \0\0\0\0".to_vec() } else { b"mBA \0\0\0\0".to_vec() };
    o.extend_from_slice(&[inputs as u8, outputs as u8, 0, 0]);
    o.extend_from_slice(&[0u8; 20]); // offsets, patched below
    let mut offs = [0u32; 5]; // B, matrix, M, CLUT, A
    let curves = |o: &mut Vec<u8>, cs: &[Curve]| {
        for c in cs {
            o.extend_from_slice(&curve_elem(c));
            pad4(o);
        }
    };
    let b = b_curves?;
    offs[0] = o.len() as u32;
    curves(&mut o, b);
    if let Some((mm, off)) = matrix {
        offs[1] = o.len() as u32;
        for v in mm.iter() {
            o.extend_from_slice(&s15(*v));
        }
        for v in off.iter() {
            o.extend_from_slice(&s15(*v));
        }
        offs[2] = o.len() as u32;
        curves(&mut o, m_curves?);
    }
    if let Some(c) = clut {
        offs[3] = o.len() as u32;
        let mut g = [0u8; 16];
        for (k, n) in c.grid.iter().enumerate() {
            g[k] = *n as u8;
        }
        o.extend_from_slice(&g);
        o.extend_from_slice(&[2, 0, 0, 0]);
        for v in &c.data {
            o.extend_from_slice(&u16q(*v).to_be_bytes());
        }
        pad4(&mut o);
        offs[4] = o.len() as u32;
        curves(&mut o, a_curves?);
    }
    for (k, v) in offs.iter().enumerate() {
        o[12 + k * 4..16 + k * 4].copy_from_slice(&v.to_be_bytes());
    }
    Some(o)
}

/// Encodes `p` as ICC v4.3 bytes (identical tag data is shared between tags). A LUT whose
/// stage layout its tag type can't store (only possible for a parsed profile whose LUT tag
/// used the other direction's type) is left out; the matrix/TRC tags still describe it.
pub fn encode(p: &Profile) -> Vec<u8> {
    let mut tags: Vec<([u8; 4], Vec<u8>)> = Vec::new();
    tags.push((*b"desc", mluc(&p.description)));
    tags.push((*b"cprt", mluc(&p.copyright)));
    tags.push((*b"wtpt", xyz_type(p.white_point)));
    if let Some(bk) = p.black_point {
        tags.push((*b"bkpt", xyz_type(bk)));
    }
    if let Some(c) = &p.chad {
        let mut o = b"sf32\0\0\0\0".to_vec();
        for v in c.iter().flatten() {
            o.extend_from_slice(&s15(*v));
        }
        tags.push((*b"chad", o));
    }
    if let Some(m) = &p.matrix {
        tags.push((*b"rXYZ", xyz_type([m[0][0], m[1][0], m[2][0]])));
        tags.push((*b"gXYZ", xyz_type([m[0][1], m[1][1], m[2][1]])));
        tags.push((*b"bXYZ", xyz_type([m[0][2], m[1][2], m[2][2]])));
    }
    if let Some([r, g, b]) = &p.trc {
        tags.push((*b"rTRC", curve_elem(r)));
        tags.push((*b"gTRC", curve_elem(g)));
        tags.push((*b"bTRC", curve_elem(b)));
    }
    if let Some(k) = &p.gray_trc {
        tags.push((*b"kTRC", curve_elem(k)));
    }
    for (luts, names, a2b) in [(&p.a2b, [b"A2B0", b"A2B1", b"A2B2"], true), (&p.b2a, [b"B2A0", b"B2A1", b"B2A2"], false)] {
        for (l, name) in luts.iter().zip(names) {
            if let Some(l) = l {
                let data = match l.kind {
                    LutKind::Lut16 => encode_mft(&l.stages, true),
                    LutKind::Lut8 => encode_mft(&l.stages, false),
                    LutKind::Ab => encode_ab(&l.stages, a2b, l.inputs, l.outputs),
                };
                if let Some(data) = data {
                    tags.push((*name, data));
                }
            }
        }
    }

    // Layout: header, tag table, then data (identical payloads share one copy).
    let table_len = 4 + tags.len() * 12;
    let mut data: Vec<u8> = Vec::new();
    let mut entries: Vec<([u8; 4], u32, u32)> = Vec::new();
    let mut placed: Vec<(usize, u32, u32)> = Vec::new(); // (tag index, offset, len)
    for (i, (sig, d)) in tags.iter().enumerate() {
        if let Some((_, off, len)) = placed.iter().find(|(j, _, _)| tags[*j].1 == *d) {
            entries.push((*sig, *off, *len));
            continue;
        }
        let off = (128 + table_len + data.len()) as u32;
        data.extend_from_slice(d);
        pad4(&mut data);
        entries.push((*sig, off, d.len() as u32));
        placed.push((i, off, d.len() as u32));
    }
    let size = 128 + table_len + data.len();
    let mut o = Vec::with_capacity(size);
    o.extend_from_slice(&(size as u32).to_be_bytes());
    o.extend_from_slice(&[0, 0, 0, 0]); // preferred CMM
    o.extend_from_slice(&[4, 0x30, 0, 0]); // version 4.3
    o.extend_from_slice(&p.class.sig().to_be_bytes());
    o.extend_from_slice(&p.color_space.sig().to_be_bytes());
    o.extend_from_slice(match p.pcs {
        Pcs::Xyz => b"XYZ ",
        Pcs::Lab => b"Lab ",
    });
    for v in [2026u16, 1, 1, 0, 0, 0] {
        o.extend_from_slice(&v.to_be_bytes());
    }
    o.extend_from_slice(b"acsp");
    o.extend_from_slice(&[0; 4]); // platform
    o.extend_from_slice(&[0; 4]); // flags
    o.extend_from_slice(&[0; 4]); // manufacturer
    o.extend_from_slice(&[0; 4]); // model
    o.extend_from_slice(&[0; 8]); // attributes
    o.extend_from_slice(&(p.rendering_intent as u32).to_be_bytes());
    for v in math::D50 {
        o.extend_from_slice(&s15(v));
    }
    o.extend_from_slice(&[0; 4]); // creator
    o.extend_from_slice(&[0; 16]); // profile id (not computed)
    o.resize(128, 0);
    o.extend_from_slice(&(entries.len() as u32).to_be_bytes());
    for (sig, off, len) in &entries {
        o.extend_from_slice(sig);
        o.extend_from_slice(&off.to_be_bytes());
        o.extend_from_slice(&len.to_be_bytes());
    }
    o.extend_from_slice(&data);
    debug_assert_eq!(o.len(), size);
    o
}
