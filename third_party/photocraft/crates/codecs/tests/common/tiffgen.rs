//! A tiny hand-rolled TIFF / BigTIFF writer for tests: any byte order, strips or tiles, any
//! planar configuration, several IFDs in a chain plus free-standing ones (SubIFDs), and access
//! to every offset so tests can corrupt the structure precisely.
#![allow(dead_code, clippy::too_many_arguments)]

/// A tag value: a number or a reference to another directory's offset (for SubIFDs).
#[derive(Clone, Copy, Debug)]
pub enum V {
    N(u64),
    Ifd(usize),
}

/// TIFF field types used by the tests.
pub const BYTE: u16 = 1;
pub const ASCII: u16 = 2;
pub const SHORT: u16 = 3;
pub const LONG: u16 = 4;
pub const RATIONAL: u16 = 5;
pub const UNDEFINED: u16 = 7;
pub const IFD: u16 = 13;
pub const LONG8: u16 = 16;
pub const IFD8: u16 = 18;

fn type_size(ty: u16) -> usize {
    match ty {
        BYTE | ASCII | UNDEFINED | 6 => 1,
        SHORT | 8 => 2,
        LONG | IFD | 9 | 11 => 4,
        RATIONAL | 10 | 12 | LONG8 | 17 | IFD8 => 8,
        _ => 1,
    }
}

#[derive(Clone, Debug, Default)]
pub struct Dir {
    /// `(tag, type, values)`; strip / tile offsets and byte counts are added from `chunks`.
    pub entries: Vec<(u16, u16, Vec<V>)>,
    /// Pixel data, one blob per strip or tile.
    pub chunks: Vec<Vec<u8>>,
    /// Tiles (322/323 must be in `entries`) instead of strips.
    pub tiled: bool,
}

impl Dir {
    pub fn tag(mut self, tag: u16, ty: u16, values: &[u64]) -> Self {
        self.entries.push((tag, ty, values.iter().map(|&v| V::N(v)).collect()));
        self
    }
    pub fn bytes(mut self, tag: u16, ty: u16, data: &[u8]) -> Self {
        self.entries.push((tag, ty, data.iter().map(|&v| V::N(u64::from(v))).collect()));
        self
    }
    pub fn sub_ifds(mut self, ty: u16, dirs: &[usize]) -> Self {
        self.entries.push((330, ty, dirs.iter().map(|&d| V::Ifd(d)).collect()));
        self
    }
}

/// A basic directory: `w`×`h`, `spp` samples of `bits` bits, the given photometric, chunky,
/// uncompressed, one strip per `rows` rows holding `data` (already in file byte order).
pub fn strips(w: u32, h: u32, bits: u16, spp: u16, photometric: u16, rows: u32, data: &[u8]) -> Dir {
    let row = (w as usize * bits as usize * spp as usize).div_ceil(8);
    let chunks = data.chunks((row * rows as usize).max(1)).map(<[u8]>::to_vec).collect();
    Dir { entries: base(w, h, bits, spp, photometric), chunks, tiled: false }.tag(278, LONG, &[u64::from(rows)])
}

pub fn base(w: u32, h: u32, bits: u16, spp: u16, photometric: u16) -> Vec<(u16, u16, Vec<V>)> {
    vec![
        (256, LONG, vec![V::N(u64::from(w))]),
        (257, LONG, vec![V::N(u64::from(h))]),
        (258, SHORT, vec![V::N(u64::from(bits)); spp as usize]),
        (259, SHORT, vec![V::N(1)]),
        (262, SHORT, vec![V::N(u64::from(photometric))]),
        (277, SHORT, vec![V::N(u64::from(spp))]),
    ]
}

/// Where things ended up in a built file.
#[derive(Debug, Default)]
pub struct Built {
    pub bytes: Vec<u8>,
    /// Offset of each directory (`dirs` order).
    pub ifd_at: Vec<u64>,
    /// Offset of each directory's next-IFD pointer field.
    pub next_at: Vec<u64>,
    /// Offset of each chunk's data, per directory.
    pub chunk_at: Vec<Vec<u64>>,
}

/// Builds a file. `chain` lists the directories of the main IFD chain in order; directories not
/// in it are only reachable through SubIFD references.
pub fn build(little: bool, big: bool, dirs: &[Dir], chain: &[usize]) -> Built {
    let e16 = |v: u16| if little { v.to_le_bytes().to_vec() } else { v.to_be_bytes().to_vec() };
    let e32 = |v: u32| if little { v.to_le_bytes().to_vec() } else { v.to_be_bytes().to_vec() };
    let e64 = |v: u64| if little { v.to_le_bytes().to_vec() } else { v.to_be_bytes().to_vec() };
    let enc = |ty: u16, v: u64| -> Vec<u8> {
        match type_size(ty) {
            1 => vec![v as u8],
            2 => e16(v as u16),
            4 => e32(v as u32),
            _ => e64(v),
        }
    };
    let header = if big { 16 } else { 8 };
    let entry_len = if big { 20 } else { 12 };
    let inline = if big { 8 } else { 4 };
    let off_ty = if big { LONG8 } else { LONG };

    // Complete entry lists (with chunk offsets / byte counts), sorted by tag.
    let mut full: Vec<Vec<(u16, u16, Vec<V>)>> = Vec::new();
    for d in dirs {
        let mut e = d.entries.clone();
        let (ot, ct) = if d.tiled { (324, 325) } else { (273, 279) };
        if !d.chunks.is_empty() || !e.iter().any(|x| x.0 == ot) {
            e.retain(|x| x.0 != ot && x.0 != ct);
            e.push((ot, off_ty, vec![V::N(0); d.chunks.len()]));
            e.push((ct, off_ty, d.chunks.iter().map(|c| V::N(c.len() as u64)).collect()));
        }
        e.sort_by_key(|x| x.0);
        full.push(e);
    }

    // Pass 1: positions. Layout: header, then per directory its chunks, its out-of-line values
    // and its table.
    let mut pos = header as u64;
    let mut ifd_at = Vec::new();
    let mut chunk_at = Vec::new();
    for (d, e) in dirs.iter().zip(&full) {
        let mut ca = Vec::new();
        for c in &d.chunks {
            ca.push(pos);
            pos += c.len() as u64;
        }
        chunk_at.push(ca);
        for (_, ty, vals) in e {
            let n = type_size(*ty) * vals.len();
            if n > inline {
                pos += n as u64;
            }
        }
        pos += pos % 2; // word alignment
        ifd_at.push(pos);
        pos += (if big { 8 } else { 2 }) + (e.len() * entry_len) as u64 + if big { 8 } else { 4 };
    }

    // Pass 2: bytes.
    let mut f = Vec::new();
    f.extend_from_slice(if little { b"II" } else { b"MM" });
    let first = chain.first().map_or(0, |&i| ifd_at[i]);
    if big {
        f.extend(e16(43));
        f.extend(e16(8));
        f.extend(e16(0));
        f.extend(e64(first));
    } else {
        f.extend(e16(42));
        f.extend(e32(first as u32));
    }
    let mut next_at = vec![0; dirs.len()];
    for (k, (d, e)) in dirs.iter().zip(&full).enumerate() {
        for c in &d.chunks {
            f.extend_from_slice(c);
        }
        let (ot, _) = if d.tiled { (324, 325) } else { (273, 279) };
        let resolve = |tag: u16, i: usize, v: V| -> u64 {
            if tag == ot && !d.chunks.is_empty() {
                return chunk_at[k][i];
            }
            match v {
                V::N(n) => n,
                V::Ifd(j) => ifd_at[j],
            }
        };
        let mut table = Vec::new();
        table.extend(if big { e64(e.len() as u64) } else { e16(e.len() as u16) });
        for (tag, ty, vals) in e {
            let mut data = Vec::new();
            for (i, v) in vals.iter().enumerate() {
                data.extend(enc(*ty, resolve(*tag, i, *v)));
            }
            table.extend(e16(*tag));
            table.extend(e16(*ty));
            table.extend(if big { e64(vals.len() as u64) } else { e32(vals.len() as u32) });
            if data.len() <= inline {
                data.resize(inline, 0);
                table.extend(data);
            } else {
                let at = f.len() as u64;
                f.extend(data);
                table.extend(if big { e64(at) } else { e32(at as u32) });
            }
        }
        if f.len() % 2 == 1 {
            f.push(0);
        }
        assert_eq!(f.len() as u64, ifd_at[k]);
        let next = chain.iter().position(|&c| c == k).and_then(|p| chain.get(p + 1)).map_or(0, |&n| ifd_at[n]);
        f.extend(table);
        next_at[k] = f.len() as u64;
        f.extend(if big { e64(next) } else { e32(next as u32) });
    }
    Built { bytes: f, ifd_at, next_at, chunk_at }
}

/// Overwrites the next-IFD pointer of directory `k` (for cycles and dangling pointers).
pub fn set_next(b: &mut Built, little: bool, big: bool, k: usize, to: u64) {
    let at = b.next_at[k] as usize;
    let v = if big {
        if little { to.to_le_bytes().to_vec() } else { to.to_be_bytes().to_vec() }
    } else if little {
        (to as u32).to_le_bytes().to_vec()
    } else {
        (to as u32).to_be_bytes().to_vec()
    };
    b.bytes[at..at + v.len()].copy_from_slice(&v);
}

/// Locates entry `tag` of directory `k`: `(entry position, type, count, value position)`.
pub fn find_entry(b: &Built, little: bool, big: bool, k: usize, tag: u16) -> Option<(usize, u16, u64, usize)> {
    let f = &b.bytes;
    let rd = |at: usize, n: usize| -> u64 {
        let s = &f[at..at + n];
        let mut v = 0u64;
        for i in 0..n {
            let byte = if little { s[n - 1 - i] } else { s[i] };
            v = (v << 8) | u64::from(byte);
        }
        v
    };
    let at = b.ifd_at[k] as usize;
    let (n, first, entry, inline) = if big { (rd(at, 8), at + 8, 20, 8) } else { (rd(at, 2), at + 2, 12, 4) };
    for i in 0..n as usize {
        let e = first + i * entry;
        if rd(e, 2) as u16 != tag {
            continue;
        }
        let ty = rd(e + 2, 2) as u16;
        let count = if big { rd(e + 4, 8) } else { rd(e + 4, 4) };
        let field = e + if big { 12 } else { 8 };
        let value_at = if count as usize * type_size(ty) <= inline { field } else { rd(field, inline) as usize };
        return Some((e, ty, count, value_at));
    }
    None
}

/// Overwrites value `i` of entry `tag` in directory `k` (in its own type and width).
pub fn patch_value(b: &mut Built, little: bool, big: bool, k: usize, tag: u16, i: usize, value: u64) {
    let (_, ty, _, at) = find_entry(b, little, big, k, tag).expect("tag present");
    let n = type_size(ty);
    let bytes = if little { value.to_le_bytes() } else { value.to_be_bytes() };
    let v = if little { bytes[..n].to_vec() } else { bytes[8 - n..].to_vec() };
    let at = at + i * n;
    b.bytes[at..at + n].copy_from_slice(&v);
}

/// Overwrites the count field of entry `tag` in directory `k`.
pub fn patch_count(b: &mut Built, little: bool, big: bool, k: usize, tag: u16, count: u64) {
    let (e, ..) = find_entry(b, little, big, k, tag).expect("tag present");
    let v = if big {
        if little { count.to_le_bytes().to_vec() } else { count.to_be_bytes().to_vec() }
    } else if little {
        (count as u32).to_le_bytes().to_vec()
    } else {
        (count as u32).to_be_bytes().to_vec()
    };
    b.bytes[e + 4..e + 4 + v.len()].copy_from_slice(&v);
}

/// Samples to file bytes in the given order.
pub fn u16s(v: &[u16], little: bool) -> Vec<u8> {
    v.iter().flat_map(|x| if little { x.to_le_bytes() } else { x.to_be_bytes() }).collect()
}
pub fn f32s(v: &[f32], little: bool) -> Vec<u8> {
    v.iter().flat_map(|x| if little { x.to_le_bytes() } else { x.to_be_bytes() }).collect()
}

/// Splits an interleaved `w`×`h` image of `spp` samples (`bps` bytes each) into tiles of
/// `tw`×`th`, padded with zeros; chunky (one blob per tile) or planar (all tiles of plane 0,
/// then plane 1, …).
pub fn tile_split(px: &[u8], w: usize, h: usize, spp: usize, bps: usize, tw: usize, th: usize, planar: bool) -> Vec<Vec<u8>> {
    let across = w.div_ceil(tw);
    let down = h.div_ceil(th);
    let planes = if planar { spp } else { 1 };
    let per = if planar { 1 } else { spp };
    let mut out = Vec::new();
    for p in 0..planes {
        for ty in 0..down {
            for tx in 0..across {
                let mut t = vec![0u8; tw * th * per * bps];
                for y in 0..th {
                    for x in 0..tw {
                        let (ix, iy) = (tx * tw + x, ty * th + y);
                        if ix >= w || iy >= h {
                            continue;
                        }
                        for s in 0..per {
                            let src = ((iy * w + ix) * spp + if planar { p } else { s }) * bps;
                            let dst = ((y * tw + x) * per + s) * bps;
                            t[dst..dst + bps].copy_from_slice(&px[src..src + bps]);
                        }
                    }
                }
                out.push(t);
            }
        }
    }
    out
}

/// Splits an interleaved image into planar strips of `rows` rows: plane 0's strips, then
/// plane 1's, ….
pub fn planar_strips(px: &[u8], w: usize, h: usize, spp: usize, bps: usize, rows: usize) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    for p in 0..spp {
        for y0 in (0..h).step_by(rows) {
            let mut s = Vec::new();
            for y in y0..(y0 + rows).min(h) {
                for x in 0..w {
                    let src = ((y * w + x) * spp + p) * bps;
                    s.extend_from_slice(&px[src..src + bps]);
                }
            }
            out.push(s);
        }
    }
    out
}
