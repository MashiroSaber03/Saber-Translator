//! Synthetic raw-file generators for tests and fuzz seeds: a lossless-JPEG
//! (T.81 process 14) encoder, a small TIFF writer, and DNG / CR2 / TIFF-EP /
//! RW2 / ORF / RAF builders. The files are written from the public specifications only and
//! are meant to exercise the decoder, not to be camera-accurate.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

// ---------------------------------------------------------------- lossless JPEG

/// A CR3 container holding one JPEG track and CMT1 orientation. It deliberately
/// contains no CRX data: used to test explicit preview fallback and BMFF bounds.
pub fn cr3_preview(jpeg: &[u8], orientation: u16) -> Vec<u8> {
    fn bx(kind: &[u8; 4], data: &[u8]) -> Vec<u8> {
        let mut b = ((data.len() + 8) as u32).to_be_bytes().to_vec();
        b.extend_from_slice(kind);
        b.extend_from_slice(data);
        b
    }
    fn moov(jpeg_len: usize, offset: u64, orientation: u16) -> Vec<u8> {
        let mut tiff = TiffBuilder::default();
        let ifd = tiff.ifd(vec![(274, Val::Short(vec![orientation]))]);
        tiff.chain = vec![ifd];
        let mut meta = vec![0x85, 0xc0, 0xb6, 0x87, 0x82, 0x0f, 0x11, 0xe0, 0x81, 0x11, 0xf4, 0xce, 0x46, 0x2b, 0x6a, 0x48];
        meta.extend(bx(b"CMT1", &tiff.build()));
        let mut entry = vec![0; 82];
        entry.extend(bx(b"JPEG", &[0; 4]));
        let mut description = vec![0, 0, 0, 0, 0, 0, 0, 1];
        description.extend(bx(b"CRAW", &entry));
        let mut sizes = vec![0; 4];
        sizes.extend_from_slice(&(jpeg_len as u32).to_be_bytes());
        sizes.extend_from_slice(&1u32.to_be_bytes());
        let mut offsets = vec![0, 0, 0, 0, 0, 0, 0, 1];
        offsets.extend_from_slice(&offset.to_be_bytes());
        let mapping: Vec<u8> = [0u32, 1, 1, 1, 1].iter().flat_map(|n| n.to_be_bytes()).collect();
        let table = [bx(b"stsd", &description), bx(b"stsz", &sizes), bx(b"co64", &offsets), bx(b"stsc", &mapping)].concat();
        let track = bx(b"trak", &bx(b"mdia", &bx(b"minf", &bx(b"stbl", &table))));
        bx(b"moov", &[bx(b"uuid", &meta), track].concat())
    }
    let ftyp = bx(b"ftyp", b"crx \0\0\0\x01crx isom");
    let header = moov(jpeg.len(), 0, orientation);
    let offset = ftyp.len() + header.len() + 8;
    [ftyp, moov(jpeg.len(), offset as u64, orientation), bx(b"mdat", jpeg)].concat()
}

/// Huffman code lengths assigned to categories in decreasing frequency
/// order. Kraft sum < 1 and no code is all ones.
const LENGTHS: [u8; 17] = [2, 3, 3, 3, 4, 4, 5, 5, 6, 6, 7, 8, 9, 10, 11, 12, 13];

fn category(d: i32) -> u32 {
    if d == 0 {
        0
    } else if d == 32768 {
        16
    } else {
        32 - d.unsigned_abs().leading_zeros()
    }
}

struct BitWriter {
    out: Vec<u8>,
    acc: u32,
    n: u32,
}

impl BitWriter {
    fn put(&mut self, v: u32, bits: u32) {
        for i in (0..bits).rev() {
            self.acc = (self.acc << 1) | ((v >> i) & 1);
            self.n += 1;
            if self.n == 8 {
                let b = self.acc as u8;
                self.out.push(b);
                if b == 0xFF {
                    self.out.push(0);
                }
                self.acc = 0;
                self.n = 0;
            }
        }
    }
    fn flush(&mut self) {
        if self.n > 0 {
            let pad = 8 - self.n;
            self.put((1 << pad) - 1, pad);
        }
    }
}

/// Encodes `samples` (raster order, `components` interleaved, values below
/// `2^precision`) as a lossless JPEG with the given predictor (1–7) and an
/// optional restart interval in rows.
pub fn lj92_encode(samples: &[u16], width: usize, height: usize, components: usize, precision: u8, predictor: u8, restart_rows: Option<usize>) -> Vec<u8> {
    assert_eq!(samples.len(), width * height * components);
    let row = width * components;
    let interval = restart_rows.unwrap_or(usize::MAX);
    // Differences per sample, following T.81 H.1.2.1 prediction rules.
    let mut diffs = Vec::with_capacity(samples.len());
    let initial = 1i32 << (precision - 1);
    for y in 0..height {
        let first_line = y % interval == 0;
        for x in 0..width {
            for c in 0..components {
                let i = y * row + x * components + c;
                let px = if x == 0 {
                    if first_line { initial } else { i32::from(samples[i - row]) }
                } else if first_line {
                    i32::from(samples[i - components])
                } else {
                    let ra = i32::from(samples[i - components]);
                    let rb = i32::from(samples[i - row]);
                    let rc = i32::from(samples[i - row - components]);
                    match predictor {
                        1 => ra,
                        2 => rb,
                        3 => rc,
                        4 => ra + rb - rc,
                        5 => ra + ((rb - rc) >> 1),
                        6 => rb + ((ra - rc) >> 1),
                        _ => (ra + rb) >> 1,
                    }
                };
                let mut d = (i32::from(samples[i]) - px) & 0xFFFF;
                if d > 32768 {
                    d -= 65536;
                }
                diffs.push(d);
            }
        }
    }
    // Table: categories ordered by frequency get increasing code lengths.
    let mut freq = [0usize; 17];
    for &d in &diffs {
        freq[category(d) as usize] += 1;
    }
    let mut order: Vec<usize> = (0..17).collect();
    order.sort_by_key(|&c| std::cmp::Reverse(freq[c]));
    let mut len_of = [0u8; 17];
    for (rank, &c) in order.iter().enumerate() {
        len_of[c] = LENGTHS[rank];
    }
    let mut counts = [0u8; 16];
    let mut vals: Vec<u8> = Vec::new();
    let mut code_of = [(0u32, 0u32); 17];
    let mut code = 0u32;
    for len in 1..=16u8 {
        for c in 0..17 {
            if len_of[c] == len {
                counts[len as usize - 1] += 1;
                vals.push(c as u8);
                code_of[c] = (code, u32::from(len));
                code += 1;
            }
        }
        code <<= 1;
    }
    let mut out = vec![0xFF, 0xD8];
    // SOF3
    out.extend_from_slice(&[0xFF, 0xC3]);
    out.extend_from_slice(&((8 + 3 * components) as u16).to_be_bytes());
    out.push(precision);
    out.extend_from_slice(&(height as u16).to_be_bytes());
    out.extend_from_slice(&(width as u16).to_be_bytes());
    out.push(components as u8);
    for c in 0..components {
        out.extend_from_slice(&[c as u8 + 1, 0x11, 0]);
    }
    // DHT (one table, id 0)
    out.extend_from_slice(&[0xFF, 0xC4]);
    out.extend_from_slice(&((2 + 17 + vals.len()) as u16).to_be_bytes());
    out.push(0x00);
    out.extend_from_slice(&counts);
    out.extend_from_slice(&vals);
    if let Some(r) = restart_rows {
        out.extend_from_slice(&[0xFF, 0xDD, 0, 4]);
        out.extend_from_slice(&((r * width) as u16).to_be_bytes());
    }
    // SOS
    out.extend_from_slice(&[0xFF, 0xDA]);
    out.extend_from_slice(&((6 + 2 * components) as u16).to_be_bytes());
    out.push(components as u8);
    for c in 0..components {
        out.extend_from_slice(&[c as u8 + 1, 0x00]);
    }
    out.extend_from_slice(&[predictor, 0, 0]);
    let mut w = BitWriter { out, acc: 0, n: 0 };
    let mut rst = 0u8;
    for (i, &d) in diffs.iter().enumerate() {
        if i > 0 && i % (interval.saturating_mul(row)) == 0 {
            w.flush();
            w.out.extend_from_slice(&[0xFF, 0xD0 + rst]);
            rst = (rst + 1) % 8;
        }
        let s = category(d);
        let (c, l) = code_of[s as usize];
        w.put(c, l);
        if (1..16).contains(&s) {
            let extra = if d < 0 { (d - 1) as u32 & ((1 << s) - 1) } else { d as u32 };
            w.put(extra, s);
        }
    }
    w.flush();
    let mut out = w.out;
    out.extend_from_slice(&[0xFF, 0xD9]);
    out
}

// ---------------------------------------------------------------- TIFF writer

/// A TIFF entry value.
#[derive(Debug, Clone)]
pub enum Val {
    Byte(Vec<u8>),
    Ascii(String),
    Short(Vec<u16>),
    Long(Vec<u32>),
    Double(Vec<f64>),
    Rational(Vec<(u32, u32)>),
    SRational(Vec<(i32, i32)>),
    Undefined(Vec<u8>),
    /// Offsets of these blobs (LONG).
    Blobs(Vec<usize>),
    /// Offsets of these IFDs (LONG), for SubIFDs / EXIF.
    Ifds(Vec<usize>),
    /// An UNDEFINED value whose bytes are this IFD (a maker note).
    IfdBytes(usize),
}

/// A TIFF file under construction.
#[derive(Debug, Clone, Default)]
pub struct TiffBuilder {
    pub big_endian: bool,
    /// Entries per IFD (sorted by tag on write).
    pub ifds: Vec<Vec<(u16, Val)>>,
    /// The main IFD chain, by index into `ifds`.
    pub chain: Vec<usize>,
    pub blobs: Vec<Vec<u8>>,
    /// Write a CR2 header pointing at this IFD as the raw IFD.
    pub cr2_raw_ifd: Option<usize>,
}

impl TiffBuilder {
    pub fn ifd(&mut self, entries: Vec<(u16, Val)>) -> usize {
        self.ifds.push(entries);
        self.ifds.len() - 1
    }

    pub fn blob(&mut self, data: Vec<u8>) -> usize {
        self.blobs.push(data);
        self.blobs.len() - 1
    }

    fn val_size(&self, v: &Val) -> usize {
        match v {
            Val::Byte(b) | Val::Undefined(b) => b.len(),
            Val::Ascii(s) => s.len() + 1,
            Val::Short(s) => 2 * s.len(),
            Val::Long(l) => 4 * l.len(),
            Val::Double(d) => 8 * d.len(),
            Val::Rational(r) => 8 * r.len(),
            Val::SRational(r) => 8 * r.len(),
            Val::Blobs(b) => 4 * b.len(),
            Val::Ifds(i) => 4 * i.len(),
            Val::IfdBytes(i) => self.ifd_size(*i),
        }
    }

    fn ifd_size(&self, i: usize) -> usize {
        2 + 12 * self.ifds[i].len() + 4
    }

    pub fn build(&self) -> Vec<u8> {
        let header = if self.cr2_raw_ifd.is_some() { 16 } else { 8 };
        // Layout: each IFD, then its overflow values; then the blobs.
        let mut pos = header;
        let mut ifd_pos = vec![0usize; self.ifds.len()];
        let mut val_pos: Vec<Vec<usize>> = vec![Vec::new(); self.ifds.len()];
        // Maker-note IFDs are written inside their owner's value area.
        let embedded: Vec<usize> = self.ifds.iter().flatten().filter_map(|(_, v)| if let Val::IfdBytes(i) = v { Some(*i) } else { None }).collect();
        for i in 0..self.ifds.len() {
            if embedded.contains(&i) {
                continue;
            }
            ifd_pos[i] = pos;
            pos += self.ifd_size(i);
            let mut vp = Vec::new();
            for (_, v) in self.sorted(i) {
                let size = self.val_size(&v);
                if size > 4 {
                    vp.push(pos);
                    if let Val::IfdBytes(e) = v {
                        ifd_pos[e] = pos;
                        // The embedded IFD's own values follow it.
                        let inner = pos + self.ifd_size(e);
                        let mut ip = Vec::new();
                        let mut q = inner;
                        for (_, iv) in self.sorted(e) {
                            let s = self.val_size(&iv);
                            if s > 4 {
                                ip.push(q);
                                q += s + (s & 1);
                            } else {
                                ip.push(0);
                            }
                        }
                        val_pos[e] = ip;
                        pos = q;
                    } else {
                        pos += size + (size & 1);
                    }
                } else {
                    vp.push(0);
                }
            }
            val_pos[i] = vp;
        }
        let mut blob_pos = Vec::new();
        for b in &self.blobs {
            blob_pos.push(pos);
            pos += b.len() + (b.len() & 1);
        }
        let mut out = vec![0u8; pos];
        let be = self.big_endian;
        let w16 = |o: &mut [u8], at: usize, v: u16| o[at..at + 2].copy_from_slice(&if be { v.to_be_bytes() } else { v.to_le_bytes() });
        let w32 = |o: &mut [u8], at: usize, v: u32| o[at..at + 4].copy_from_slice(&if be { v.to_be_bytes() } else { v.to_le_bytes() });
        out[0..4].copy_from_slice(if be { b"MM\0*" } else { b"II*\0" });
        w32(&mut out, 4, ifd_pos[self.chain[0]] as u32);
        if let Some(r) = self.cr2_raw_ifd {
            out[8..12].copy_from_slice(b"CR\x02\0");
            w32(&mut out, 12, ifd_pos[r] as u32);
        }
        for i in 0..self.ifds.len() {
            let at = ifd_pos[i];
            let entries = self.sorted(i);
            w16(&mut out, at, entries.len() as u16);
            for (k, (tag, v)) in entries.iter().enumerate() {
                let e = at + 2 + 12 * k;
                let (typ, count, bytes) = self.encode(v, &blob_pos, &ifd_pos);
                w16(&mut out, e, *tag);
                w16(&mut out, e + 2, typ);
                w32(&mut out, e + 4, count);
                if bytes.len() <= 4 {
                    out[e + 8..e + 8 + bytes.len()].copy_from_slice(&bytes);
                } else {
                    let vp = val_pos[i][k];
                    w32(&mut out, e + 8, vp as u32);
                    if !matches!(v, Val::IfdBytes(_)) {
                        out[vp..vp + bytes.len()].copy_from_slice(&bytes);
                    }
                }
            }
            let next = self.chain.iter().position(|&c| c == i).and_then(|p| self.chain.get(p + 1)).map(|&n| ifd_pos[n]).unwrap_or(0);
            w32(&mut out, at + 2 + 12 * entries.len(), next as u32);
        }
        for (b, &p) in self.blobs.iter().zip(&blob_pos) {
            out[p..p + b.len()].copy_from_slice(b);
        }
        out
    }

    fn sorted(&self, i: usize) -> Vec<(u16, Val)> {
        let mut e = self.ifds[i].clone();
        e.sort_by_key(|(t, _)| *t);
        e
    }

    fn encode(&self, v: &Val, blob_pos: &[usize], ifd_pos: &[usize]) -> (u16, u32, Vec<u8>) {
        let be = self.big_endian;
        let b16 = |v: u16| if be { v.to_be_bytes() } else { v.to_le_bytes() };
        let b32 = |v: u32| if be { v.to_be_bytes() } else { v.to_le_bytes() };
        let b64 = |v: u64| if be { v.to_be_bytes() } else { v.to_le_bytes() };
        match v {
            Val::Byte(b) => (1, b.len() as u32, b.clone()),
            Val::Undefined(b) => (7, b.len() as u32, b.clone()),
            Val::Ascii(s) => {
                let mut b = s.as_bytes().to_vec();
                b.push(0);
                (2, b.len() as u32, b)
            }
            Val::Short(s) => (3, s.len() as u32, s.iter().flat_map(|&v| b16(v)).collect()),
            Val::Long(l) => (4, l.len() as u32, l.iter().flat_map(|&v| b32(v)).collect()),
            Val::Double(d) => (12, d.len() as u32, d.iter().flat_map(|v| b64(v.to_bits())).collect()),
            Val::Rational(r) => (5, r.len() as u32, r.iter().flat_map(|&(n, d)| [b32(n), b32(d)].concat()).collect()),
            Val::SRational(r) => (10, r.len() as u32, r.iter().flat_map(|&(n, d)| [b32(n as u32), b32(d as u32)].concat()).collect()),
            Val::Blobs(b) => (4, b.len() as u32, b.iter().flat_map(|&i| b32(blob_pos[i] as u32)).collect()),
            Val::Ifds(i) => (4, i.len() as u32, i.iter().flat_map(|&k| b32(ifd_pos[k] as u32)).collect()),
            Val::IfdBytes(i) => (7, self.ifd_size(*i) as u32, vec![0; 8]),
        }
    }
}

fn srat(v: f64) -> (i32, i32) {
    ((v * 10000.0).round() as i32, 10000)
}

fn urat(v: f64) -> (u32, u32) {
    ((v * 10000.0).round() as u32, 10000)
}

// ---------------------------------------------------------------- DNG

/// How the DNG stores its raw data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DngStorage {
    /// One strip per `rows` rows, uncompressed at `bits` per sample (8, 12, 16…).
    Strips { rows: usize },
    /// Adobe DNG compression 34892: baseline JPEG tiles, 8-bit luma at tile size (the reader
    /// decodes and widens to 16 bits). Quality 60 leaves visible-but-mild artifacts.
    LossyJpegTiles { width: usize, height: usize },
    /// Adobe DNG compression 8: zlib streams with the TIFF predictor (1 = none, 2 = horizontal
    /// differencing) applied before compressing.
    DeflateStrips { rows: usize, predictor: u16 },
    /// Lossless-JPEG tiles of this size (2 components per JPEG row, as Adobe writes CFA tiles).
    Lj92Tiles { width: usize, height: usize },
    /// Lossless-JPEG strips of this many rows, 1 component.
    Lj92Strips { rows: usize },
    /// JPEG XL strips of this many rows (Compression 52546): one codestream per strip, taken from
    /// [`DngSpec::jxl_segments`] (the tests encode them; this crate has no JPEG XL encoder).
    JxlStrips { rows: usize },
    /// JPEG XL tiles of this size, one codestream per tile in row-major order from
    /// [`DngSpec::jxl_segments`].
    JxlTiles { width: usize, height: usize },
}

/// A synthetic DNG.
#[derive(Debug, Clone)]
pub struct DngSpec {
    pub width: usize,
    pub height: usize,
    /// 1 (CFA, or monochrome LinearRaw with `monochrome`) or 3 (LinearRaw).
    pub samples: usize,
    /// With `samples` 1: a monochrome LinearRaw image (no CFA tags) instead of CFA data.
    pub monochrome: bool,
    /// Row-major samples (`width * height * samples`).
    pub data: Vec<u16>,
    pub bits: u16,
    pub cfa: [u8; 4],
    pub storage: DngStorage,
    pub big_endian: bool,
    pub black: Vec<u32>,
    pub black_repeat: (u16, u16),
    pub white: u32,
    /// top, left, bottom, right
    pub active_area: Option<[u32; 4]>,
    /// origin (x, y), size (w, h), relative to the active area
    pub default_crop: Option<([u32; 2], [u32; 2])>,
    /// Double-precision origin override for malformed-metadata regression tests.
    pub default_crop_origin_double: Option<[f64; 2]>,
    /// A DefaultCropSize with no DefaultCropOrigin (`default_crop` must be `None`).
    pub default_crop_size_only: Option<[u32; 2]>,
    pub linearization: Option<Vec<u16>>,
    /// (illuminant code, ColorMatrix row-major)
    pub color_matrix1: Option<(u16, [f64; 9])>,
    pub color_matrix2: Option<(u16, [f64; 9])>,
    pub forward_matrix1: Option<[f64; 9]>,
    pub forward_matrix2: Option<[f64; 9]>,
    pub as_shot_neutral: Option<[f64; 3]>,
    pub baseline_exposure: Option<f64>,
    /// Raw OpcodeList2 bytes (see [`gain_map_opcode_list`]).
    pub opcode_list2: Option<Vec<u8>>,
    /// Encoded segments for [`DngStorage::JxlStrips`] / [`DngStorage::JxlTiles`].
    pub jxl_segments: Vec<Vec<u8>>,
    /// Raw ProfileGainTableMap bytes, written to IFD0 (contents are not interpreted).
    pub profile_gain_table_map: Option<Vec<u8>>,
    pub orientation: u16,
    pub make: String,
    pub model: String,
    /// Extra IFD0 entries, written after the generated ones (e.g. malformed colour tags).
    pub ifd0_extra: Vec<(u16, Val)>,
}

impl DngSpec {
    /// A CFA DNG with default metadata around `data` (RGGB, 16-bit, one strip).
    pub fn cfa(width: usize, height: usize, data: Vec<u16>) -> Self {
        DngSpec {
            width,
            height,
            samples: 1,
            monochrome: false,
            data,
            bits: 16,
            cfa: [0, 1, 1, 2],
            storage: DngStorage::Strips { rows: height },
            big_endian: false,
            black: vec![0],
            black_repeat: (1, 1),
            white: 65535,
            active_area: None,
            default_crop: None,
            default_crop_origin_double: None,
            default_crop_size_only: None,
            linearization: None,
            color_matrix1: None,
            color_matrix2: None,
            forward_matrix1: None,
            forward_matrix2: None,
            as_shot_neutral: None,
            baseline_exposure: None,
            orientation: 1,
            opcode_list2: None,
            jxl_segments: Vec::new(),
            profile_gain_table_map: None,
            make: "Photocraft".into(),
            model: "Synthetic".into(),
            ifd0_extra: Vec::new(),
        }
    }

    /// A monochrome (one-sample LinearRaw) DNG around `data`, as monochrome cameras and film
    /// scanners write them: no CFA tags and no colour matrices.
    pub fn monochrome(width: usize, height: usize, data: Vec<u16>) -> Self {
        DngSpec { monochrome: true, ..Self::cfa(width, height, data) }
    }

    fn is_cfa(&self) -> bool {
        self.samples == 1 && !self.monochrome
    }

    fn pack(&self, rows: std::ops::Range<usize>) -> Vec<u8> {
        let n = self.width * self.samples;
        let vals = &self.data[rows.start * n..rows.end * n];
        match self.bits {
            8 => vals.iter().map(|&v| v as u8).collect(),
            16 => vals.iter().flat_map(|&v| if self.big_endian { v.to_be_bytes() } else { v.to_le_bytes() }).collect(),
            b => {
                // MSB-first bit packing, each row padded to a byte.
                let mut out = Vec::new();
                for row in vals.chunks(n) {
                    let mut acc: u64 = 0;
                    let mut have = 0u32;
                    for &v in row {
                        acc = (acc << b) | u64::from(v);
                        have += u32::from(b);
                        while have >= 8 {
                            have -= 8;
                            out.push((acc >> have) as u8);
                        }
                    }
                    if have > 0 {
                        out.push((acc << (8 - have)) as u8);
                    }
                }
                out
            }
        }
    }

    pub fn build(&self) -> Vec<u8> {
        let mut t = TiffBuilder { big_endian: self.big_endian, ..Default::default() };
        let mut raw: Vec<(u16, Val)> = vec![
            (254, Val::Long(vec![0])),
            (256, Val::Long(vec![self.width as u32])),
            (257, Val::Long(vec![self.height as u32])),
            (258, Val::Short(vec![self.bits; self.samples])),
            (262, Val::Short(vec![if self.is_cfa() { 32803 } else { 34892 }])),
            (277, Val::Short(vec![self.samples as u16])),
            (284, Val::Short(vec![1])),
            (50717, Val::Long(vec![self.white; self.samples])),
            (50713, Val::Short(vec![self.black_repeat.0, self.black_repeat.1])),
            (50714, Val::Long(self.black.clone())),
        ];
        if self.is_cfa() {
            raw.push((33421, Val::Short(vec![2, 2])));
            raw.push((33422, Val::Byte(self.cfa.to_vec())));
            raw.push((50710, Val::Byte(vec![0, 1, 2])));
            raw.push((50711, Val::Short(vec![1])));
        }
        match self.storage {
            DngStorage::DeflateStrips { rows, predictor } => {
                use std::io::Write;
                let mut offs = Vec::new();
                let mut lens = Vec::new();
                for y in (0..self.height).step_by(rows) {
                    let r = (y + rows).min(self.height) - y;
                    let n = self.width * self.samples;
                    let mut v: Vec<u16> = self.data[y * n..(y + r) * n].to_vec();
                    if predictor == 2 {
                        // Horizontal differencing on the sample values, before byte packing
                        // (the reader undoes predictor after unpacking, per row, per channel).
                        let plane = if self.samples.is_multiple_of(3) { 3 } else { 1 };
                        for row in v.chunks_mut(n) {
                            for i in (plane..row.len()).rev() {
                                row[i] = row[i].wrapping_sub(row[i - plane]);
                            }
                        }
                    }
                    // Pack the delta'd values with the file's byte order and bit depth.
                    let bytes: Vec<u8> = match self.bits {
                        8 => v.iter().map(|&x| x as u8).collect(),
                        16 => v.iter().flat_map(|&x| if self.big_endian { x.to_be_bytes() } else { x.to_le_bytes() }).collect(),
                        _ => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
                    };
                    let mut e = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
                    e.write_all(&bytes).expect("zlib");
                    let z = e.finish().expect("zlib");
                    lens.push(z.len() as u32);
                    offs.push(t.blob(z));
                }
                raw.push((259, Val::Short(vec![8])));
                raw.push((317, Val::Short(vec![predictor])));
                raw.push((278, Val::Long(vec![rows as u32])));
                raw.push((273, Val::Blobs(offs)));
                raw.push((279, Val::Long(lens)));
            }
            DngStorage::Strips { rows } => {
                let mut offs = Vec::new();
                let mut lens = Vec::new();
                for y in (0..self.height).step_by(rows) {
                    let b = self.pack(y..(y + rows).min(self.height));
                    lens.push(b.len() as u32);
                    offs.push(t.blob(b));
                }
                raw.push((259, Val::Short(vec![1])));
                raw.push((278, Val::Long(vec![rows as u32])));
                raw.push((273, Val::Blobs(offs)));
                raw.push((279, Val::Long(lens)));
            }
            DngStorage::LossyJpegTiles { width: tw, height: th } => {
                let mut offs = Vec::new();
                let mut lens = Vec::new();
                let s = self.samples;
                for ty in (0..self.height).step_by(th) {
                    for tx in (0..self.width).step_by(tw) {
                        // Edge tiles are padded by repeating the last row / column. A 3-sample
                        // (LinearRaw) spec writes RGB tiles, a CFA spec gray ones.
                        let mut tile = vec![0u8; tw * th * s];
                        for y in 0..th {
                            for x in 0..tw {
                                let sy = (ty + y).min(self.height - 1);
                                let sx = (tx + x).min(self.width - 1);
                                for c in 0..s {
                                    tile[(y * tw + x) * s + c] = (self.data[(sy * self.width + sx) * s + c] >> 8) as u8;
                                }
                            }
                        }
                        let colour = if s == 3 { image::ExtendedColorType::Rgb8 } else { image::ExtendedColorType::L8 };
                        let mut j = Vec::new();
                        let enc = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut j, 60);
                        use image::ImageEncoder as _;
                        enc.write_image(&tile, tw as u32, th as u32, colour).expect("jpeg tile");
                        lens.push(j.len() as u32);
                        offs.push(t.blob(j));
                    }
                }
                raw.push((259, Val::Short(vec![34892])));
                raw.push((322, Val::Long(vec![tw as u32])));
                raw.push((323, Val::Long(vec![th as u32])));
                raw.push((324, Val::Blobs(offs)));
                raw.push((325, Val::Long(lens)));
            }
            DngStorage::Lj92Strips { rows } => {
                let mut offs = Vec::new();
                let mut lens = Vec::new();
                for y in (0..self.height).step_by(rows) {
                    let r = (y + rows).min(self.height) - y;
                    let n = self.width * self.samples;
                    let j = lj92_encode(&self.data[y * n..(y + r) * n], self.width, r, self.samples, self.bits as u8, 1, None);
                    lens.push(j.len() as u32);
                    offs.push(t.blob(j));
                }
                raw.push((259, Val::Short(vec![7])));
                raw.push((278, Val::Long(vec![rows as u32])));
                raw.push((273, Val::Blobs(offs)));
                raw.push((279, Val::Long(lens)));
            }
            DngStorage::Lj92Tiles { width: tw, height: th } => {
                let mut offs = Vec::new();
                let mut lens = Vec::new();
                let s = self.samples;
                for ty in (0..self.height).step_by(th) {
                    for tx in (0..self.width).step_by(tw) {
                        // Edge tiles are padded by repeating the last row / column.
                        let mut tile = vec![0u16; tw * th * s];
                        for y in 0..th {
                            for x in 0..tw {
                                let sy = (ty + y).min(self.height - 1);
                                let sx = (tx + x).min(self.width - 1);
                                for c in 0..s {
                                    tile[(y * tw + x) * s + c] = self.data[(sy * self.width + sx) * s + c];
                                }
                            }
                        }
                        // CFA tiles as Adobe writes them: half width, 2 components.
                        let j = if s == 1 && tw % 2 == 0 {
                            lj92_encode(&tile, tw / 2, th, 2, self.bits as u8, 1, None)
                        } else {
                            lj92_encode(&tile, tw, th, s, self.bits as u8, 1, None)
                        };
                        lens.push(j.len() as u32);
                        offs.push(t.blob(j));
                    }
                }
                raw.push((259, Val::Short(vec![7])));
                raw.push((322, Val::Long(vec![tw as u32])));
                raw.push((323, Val::Long(vec![th as u32])));
                raw.push((324, Val::Blobs(offs)));
                raw.push((325, Val::Long(lens)));
            }
            DngStorage::JxlStrips { rows } => {
                let lens = self.jxl_segments.iter().map(|j| j.len() as u32).collect();
                let offs = self.jxl_segments.iter().map(|j| t.blob(j.clone())).collect();
                raw.push((259, Val::Short(vec![52546])));
                raw.push((278, Val::Long(vec![rows as u32])));
                raw.push((273, Val::Blobs(offs)));
                raw.push((279, Val::Long(lens)));
            }
            DngStorage::JxlTiles { width: tw, height: th } => {
                let lens = self.jxl_segments.iter().map(|j| j.len() as u32).collect();
                let offs = self.jxl_segments.iter().map(|j| t.blob(j.clone())).collect();
                raw.push((259, Val::Short(vec![52546])));
                raw.push((322, Val::Long(vec![tw as u32])));
                raw.push((323, Val::Long(vec![th as u32])));
                raw.push((324, Val::Blobs(offs)));
                raw.push((325, Val::Long(lens)));
            }
        }
        if let Some(a) = self.active_area {
            raw.push((50829, Val::Long(a.to_vec())));
        }
        if let Some((o, s)) = self.default_crop {
            let origin = self.default_crop_origin_double.map(|d| Val::Double(d.to_vec())).unwrap_or_else(|| Val::Long(o.to_vec()));
            raw.push((50719, origin));
            raw.push((50720, Val::Long(s.to_vec())));
        } else if let Some(origin) = self.default_crop_origin_double {
            raw.push((50719, Val::Double(origin.to_vec())));
        } else if let Some(size) = self.default_crop_size_only {
            raw.push((50720, Val::Long(size.to_vec())));
        }
        if let Some(o) = &self.opcode_list2 {
            raw.push((51009, Val::Undefined(o.clone())));
        }
        if let Some(l) = &self.linearization {
            raw.push((50712, Val::Short(l.clone())));
        }
        let raw_ifd = t.ifd(raw);
        // IFD0: a tiny RGB thumbnail plus the DNG metadata.
        let thumb = t.blob(vec![128; 3 * 4]);
        let mut ifd0: Vec<(u16, Val)> = vec![
            (254, Val::Long(vec![1])),
            (256, Val::Long(vec![2])),
            (257, Val::Long(vec![2])),
            (258, Val::Short(vec![8, 8, 8])),
            (259, Val::Short(vec![1])),
            (262, Val::Short(vec![2])),
            (271, Val::Ascii(self.make.clone())),
            (272, Val::Ascii(self.model.clone())),
            (273, Val::Blobs(vec![thumb])),
            (274, Val::Short(vec![self.orientation])),
            (277, Val::Short(vec![3])),
            (278, Val::Long(vec![2])),
            (279, Val::Long(vec![12])),
            (330, Val::Ifds(vec![raw_ifd])),
            (50706, Val::Byte(vec![1, 4, 0, 0])),
            (50708, Val::Ascii(format!("{} {}", self.make, self.model))),
        ];
        let m = |v: &[f64; 9]| Val::SRational(v.iter().map(|&x| srat(x)).collect());
        if let Some((ill, cm)) = &self.color_matrix1 {
            ifd0.push((50721, m(cm)));
            ifd0.push((50778, Val::Short(vec![*ill])));
        }
        if let Some((ill, cm)) = &self.color_matrix2 {
            ifd0.push((50722, m(cm)));
            ifd0.push((50779, Val::Short(vec![*ill])));
        }
        if let Some(f) = &self.forward_matrix1 {
            ifd0.push((50964, m(f)));
        }
        if let Some(f) = &self.forward_matrix2 {
            ifd0.push((50965, m(f)));
        }
        if let Some(n) = self.as_shot_neutral {
            ifd0.push((50728, Val::Rational(n.iter().map(|&v| urat(v)).collect())));
        }
        if let Some(b) = self.baseline_exposure {
            ifd0.push((50730, Val::SRational(vec![srat(b)])));
        }
        if let Some(m) = &self.profile_gain_table_map {
            ifd0.push((52525, Val::Undefined(m.clone())));
        }
        ifd0.extend(self.ifd0_extra.iter().cloned());
        let i0 = t.ifd(ifd0);
        t.chain = vec![i0];
        t.build()
    }
}

/// An OpcodeList2 with one GainMap over active-area rectangle `area` (top,
/// left, bottom, right), every `pitch` rows and columns, from a 2×2 grid of
/// gains spanning the image (relative coordinates 0..1).
pub fn gain_map_opcode_list(area: [u32; 4], pitch: u32, gains: [[f32; 2]; 2]) -> Vec<u8> {
    let mut p = Vec::new();
    for v in [area[0], area[1], area[2], area[3], 0, 1, pitch, pitch, 2, 2] {
        p.extend_from_slice(&v.to_be_bytes());
    }
    for v in [1.0f64, 1.0, 0.0, 0.0] {
        p.extend_from_slice(&v.to_be_bytes());
    }
    p.extend_from_slice(&1u32.to_be_bytes());
    for g in gains.iter().flatten() {
        p.extend_from_slice(&g.to_bits().to_be_bytes());
    }
    let mut b = 1u32.to_be_bytes().to_vec();
    for v in [9u32, 0x0103_0000, 1, p.len() as u32] {
        b.extend_from_slice(&v.to_be_bytes());
    }
    b.extend_from_slice(&p);
    b
}

// ---------------------------------------------------------------- CR2

/// A synthetic CR2: one lossless JPEG with `components` components, cut into
/// vertical slices, plus a Canon maker note with SensorInfo and ColorData.
#[derive(Debug, Clone)]
pub struct Cr2Spec {
    pub width: usize,
    pub height: usize,
    /// Sensor samples, row-major.
    pub data: Vec<u16>,
    pub precision: u8,
    pub components: usize,
    /// Slice widths (all but the last equal); empty = no slice tag.
    pub slices: Vec<usize>,
    /// Image borders: left, top, right, bottom (inclusive), as in SensorInfo.
    pub borders: Option<[u16; 4]>,
    /// As-shot RGGB levels written to ColorData at word offset 0x47 (the PowerShot version).
    pub wb_rggb: Option<[u16; 4]>,
    pub orientation: u16,
    /// Canon model ID written to MakerNote tag 0x0010.
    pub model_id: Option<u32>,
}

impl Cr2Spec {
    pub fn build(&self) -> Vec<u8> {
        let (w, h) = (self.width, self.height);
        // Slice the sensor data into the JPEG sample order.
        let widths: Vec<usize> = if self.slices.is_empty() { vec![w] } else { self.slices.clone() };
        assert_eq!(widths.iter().sum::<usize>(), w);
        let mut seq = Vec::with_capacity(w * h);
        let mut x0 = 0;
        for &sw in &widths {
            for y in 0..h {
                seq.extend_from_slice(&self.data[y * w + x0..y * w + x0 + sw]);
            }
            x0 += sw;
        }
        let c = self.components;
        assert_eq!(w % c, 0);
        let jpeg = lj92_encode(&seq, w / c, h, c, self.precision, 1, None);
        let mut t = TiffBuilder::default();
        let jb = t.blob(jpeg.clone());
        let mut raw = vec![(259, Val::Short(vec![6])), (273, Val::Blobs(vec![jb])), (279, Val::Long(vec![jpeg.len() as u32]))];
        if !self.slices.is_empty() {
            let n = widths.len() - 1;
            raw.push((50752, Val::Short(vec![n as u16, widths[0] as u16, widths[n] as u16])));
        }
        let raw_ifd = t.ifd(raw);
        let mut mn: Vec<(u16, Val)> = Vec::new();
        if let Some(id) = self.model_id {
            mn.push((0x0010, Val::Long(vec![id])));
        }
        if let Some([l, tp, r, b]) = self.borders {
            mn.push((0x00E0, Val::Short(vec![34, w as u16, h as u16, 0, 0, l, tp, r, b, 0, 0, 0, 0, 0, 0, 0, 0])));
        }
        if let Some(wb) = self.wb_rggb {
            // Canon writes ColorData (0x4001) as UNDEFINED bytes that are 16-bit words; the
            // WB_RGGBLevelsAsShot of the PowerShot version sits at word offset 0x47 (the EOS
            // versions use 0x3F — see as_shot_wb).
            let mut cd = vec![0u8; 2 * 1273];
            for (i, v) in wb.iter().enumerate() {
                cd[2 * (0x47 + i)..2 * (0x47 + i) + 2].copy_from_slice(&v.to_le_bytes());
            }
            mn.push((0x4001, Val::Undefined(cd)));
        }
        let mut exif: Vec<(u16, Val)> = vec![(33434, Val::Rational(vec![(1, 100)]))];
        if !mn.is_empty() {
            let mi = t.ifd(mn);
            exif.push((37500, Val::IfdBytes(mi)));
        }
        let exif_ifd = t.ifd(exif);
        let thumb = t.blob(vec![128; 12]);
        let ifd0 = t.ifd(vec![
            (256, Val::Long(vec![2])),
            (257, Val::Long(vec![2])),
            (259, Val::Short(vec![1])),
            (262, Val::Short(vec![2])),
            (271, Val::Ascii("Canon".into())),
            (272, Val::Ascii("Canon EOS Synthetic".into())),
            (273, Val::Blobs(vec![thumb])),
            (274, Val::Short(vec![self.orientation])),
            (277, Val::Short(vec![3])),
            (279, Val::Long(vec![12])),
            (34665, Val::Ifds(vec![exif_ifd])),
        ]);
        let ifd1 = t.ifd(vec![(256, Val::Long(vec![1]))]);
        let ifd2 = t.ifd(vec![(256, Val::Long(vec![1]))]);
        t.chain = vec![ifd0, ifd1, ifd2, raw_ifd];
        t.cr2_raw_ifd = Some(raw_ifd);
        t.build()
    }
}

// ---------------------------------------------------------------- TIFF/EP

/// A synthetic uncompressed TIFF/EP raw (NEF / ARW-like): IFD0 with Make, the
/// CFA image in a SubIFD with 16-bit samples.
pub fn tiff_ep(make: &str, width: usize, height: usize, data: &[u16], cfa: [u8; 4], bits: u16, extra_raw_tags: Vec<(u16, Val)>) -> Vec<u8> {
    let mut t = TiffBuilder::default();
    let strip = t.blob(data.iter().flat_map(|v| v.to_le_bytes()).collect());
    let mut raw = vec![
        (254, Val::Long(vec![0])),
        (256, Val::Long(vec![width as u32])),
        (257, Val::Long(vec![height as u32])),
        (258, Val::Short(vec![bits])),
        (259, Val::Short(vec![1])),
        (262, Val::Short(vec![32803])),
        (273, Val::Blobs(vec![strip])),
        (277, Val::Short(vec![1])),
        (278, Val::Long(vec![height as u32])),
        (279, Val::Long(vec![(data.len() * 2) as u32])),
        (33421, Val::Short(vec![2, 2])),
        (33422, Val::Byte(cfa.to_vec())),
    ];
    raw.extend(extra_raw_tags);
    let raw_ifd = t.ifd(raw);
    let thumb = t.blob(vec![128; 12]);
    let ifd0 = t.ifd(vec![
        (254, Val::Long(vec![1])),
        (256, Val::Long(vec![2])),
        (257, Val::Long(vec![2])),
        (259, Val::Short(vec![1])),
        (262, Val::Short(vec![2])),
        (271, Val::Ascii(make.into())),
        (273, Val::Blobs(vec![thumb])),
        (277, Val::Short(vec![3])),
        (279, Val::Long(vec![12])),
        (330, Val::Ifds(vec![raw_ifd])),
    ]);
    t.chain = vec![ifd0];
    t.build()
}

/// A synthetic Nikon NEF with a Huffman-compressed (34713) strip: IFD0 with
/// Make and an EXIF IFD whose maker note (`Nikon\0`, version 2.10, then an
/// embedded TIFF in the given byte order) holds `note_tags` (for example
/// `0x0096`, the decode table, and `0x003d`, BlackLevel).
pub fn nef_compressed(width: usize, height: usize, bits: u16, strip: Vec<u8>, cfa: [u8; 4], note_big_endian: bool, note_tags: Vec<(u16, Val)>) -> Vec<u8> {
    let mut note = TiffBuilder { big_endian: note_big_endian, ..Default::default() };
    let n = note.ifd(note_tags);
    note.chain = vec![n];
    let mut maker = b"Nikon\0\x02\x10\0\0".to_vec();
    maker.extend(note.build());
    let mut t = TiffBuilder::default();
    let len = strip.len();
    let strip = t.blob(strip);
    let raw_ifd = t.ifd(vec![
        (254, Val::Long(vec![0])),
        (256, Val::Long(vec![width as u32])),
        (257, Val::Long(vec![height as u32])),
        (258, Val::Short(vec![bits])),
        (259, Val::Short(vec![34713])),
        (262, Val::Short(vec![32803])),
        (273, Val::Blobs(vec![strip])),
        (277, Val::Short(vec![1])),
        (278, Val::Long(vec![height as u32])),
        (279, Val::Long(vec![len as u32])),
        (33421, Val::Short(vec![2, 2])),
        (33422, Val::Byte(cfa.to_vec())),
    ]);
    let exif = t.ifd(vec![(37500, Val::Undefined(maker))]);
    let ifd0 = t.ifd(vec![(271, Val::Ascii("NIKON CORPORATION".into())), (330, Val::Ifds(vec![raw_ifd])), (34665, Val::Ifds(vec![exif]))]);
    t.chain = vec![ifd0];
    t.build()
}

// ---------------------------------------------------------------- Sony cRAW

/// Encodes 16 same-colour 11-bit codes as one cRAW block (see the decoder in
/// `sony.rs`): maximum, minimum, their positions, then 7-bit deltas at the
/// smallest power-of-two step that covers the range (rounded down).
pub fn craw_block(codes: &[u16; 16]) -> [u8; 16] {
    let max = *codes.iter().max().unwrap();
    let min = *codes.iter().min().unwrap();
    let imax = codes.iter().position(|&c| c == max).unwrap();
    let mut imin = codes.iter().position(|&c| c == min).unwrap();
    if imin == imax {
        imin = (imax + 1) % 16;
    }
    let mut sh = 0;
    while (128u16 << sh) <= max - min {
        sh += 1;
    }
    let mut v: u128 = u128::from(max) | u128::from(min) << 11 | (imax as u128) << 22 | (imin as u128) << 26;
    let mut at = 30;
    for (i, &c) in codes.iter().enumerate() {
        if i != imax && i != imin {
            v |= u128::from(((c - min) >> sh).min(127)) << at;
            at += 7;
        }
    }
    v.to_le_bytes()
}

/// Encodes a mosaic of 11-bit codes (`width` a multiple of 32) as cRAW rows.
pub fn craw_encode(codes: &[u16], width: usize, height: usize) -> Vec<u8> {
    assert_eq!(width % 32, 0);
    let mut out = Vec::with_capacity(width * height);
    for row in codes.chunks_exact(width).take(height) {
        for g in row.as_chunks::<32>().0 {
            for parity in 0..2 {
                let seq: [u16; 16] = std::array::from_fn(|i| g[2 * i + parity].min(2047));
                out.extend_from_slice(&craw_block(&seq));
            }
        }
    }
    out
}

/// A synthetic Sony compressed ARW: IFD0 with Make, the cRAW image in a
/// SubIFD with SonyRawFileType 2, the tone curve (0x7010), black (0x7310),
/// white (50717) and WB_RGGBLevels (0x7313).
pub fn sony_craw(width: usize, height: usize, codes: &[u16], curve: [u16; 4]) -> Vec<u8> {
    let bytes = craw_encode(codes, width, height);
    let mut t = TiffBuilder::default();
    let n = bytes.len();
    let strip = t.blob(bytes);
    let raw_ifd = t.ifd(vec![
        (254, Val::Long(vec![0])),
        (256, Val::Long(vec![width as u32])),
        (257, Val::Long(vec![height as u32])),
        (258, Val::Short(vec![12])),
        (259, Val::Short(vec![32767])),
        (262, Val::Short(vec![32803])),
        (273, Val::Blobs(vec![strip])),
        (277, Val::Short(vec![1])),
        (278, Val::Long(vec![height as u32])),
        (279, Val::Long(vec![n as u32])),
        (33421, Val::Short(vec![2, 2])),
        (33422, Val::Byte(vec![0, 1, 1, 2])),
        (0x7000, Val::Short(vec![2])),
        (0x7010, Val::Short(curve.to_vec())),
        (0x7310, Val::Short(vec![512; 4])),
        (0x7313, Val::Short(vec![2048, 1024, 1024, 1536])),
        (50717, Val::Short(vec![16383])),
    ]);
    let ifd0 = t.ifd(vec![(271, Val::Ascii("SONY".into())), (272, Val::Ascii("ILCE-Synthetic".into())), (330, Val::Ifds(vec![raw_ifd]))]);
    t.chain = vec![ifd0];
    t.build()
}

// ---------------------------------------------------------------- Panasonic RW2

/// Page size and split point of the RW2 stream (see `rw2.rs`).
const RW2_PAGE: usize = 0x4000;
const RW2_SPLIT: usize = 0x1FF8;

/// Packs `bits`-bit samples (12: 10 per block, 14: 9 per block) into the RW2
/// RawFormat 5 layout: 16-byte little-endian blocks, stored in 0x4000-byte
/// pages split at 0x1FF8.
pub fn rw2_pack(data: &[u16], width: usize, bits: u32) -> Vec<u8> {
    let ppb = if bits == 14 { 9 } else { 10 };
    assert_eq!(width % ppb, 0);
    let mut logical = Vec::new();
    for px in data.chunks_exact(ppb) {
        let mut v: u128 = 0;
        for (i, &p) in px.iter().enumerate() {
            v |= u128::from(p & ((1 << bits) - 1)) << (bits as usize * i);
        }
        logical.extend_from_slice(&v.to_le_bytes());
    }
    logical.resize(logical.len().div_ceil(RW2_PAGE) * RW2_PAGE, 0);
    let mut out = vec![0u8; logical.len()];
    for (i, &b) in logical.iter().enumerate() {
        let page = i - i % RW2_PAGE;
        out[page + (i % RW2_PAGE + RW2_SPLIT) % RW2_PAGE] = b;
    }
    out
}

/// A synthetic Panasonic RW2 (RawFormat 5) with an RGGB sensor, a 2-pixel
/// border on every side, black levels 128 / 129 / 130 and WB levels
/// 512 / 256 / 384.
pub fn rw2(width: usize, height: usize, data: &[u16], bits: u32) -> Vec<u8> {
    let mut t = TiffBuilder::default();
    let raw = t.blob(rw2_pack(data, width, bits));
    let white = ((1u32 << bits) - 1) as u16;
    let ifd0 = t.ifd(vec![
        (0x0002, Val::Short(vec![width as u16])),
        (0x0003, Val::Short(vec![height as u16])),
        (0x0004, Val::Short(vec![2])),
        (0x0005, Val::Short(vec![2])),
        (0x0006, Val::Short(vec![height as u16 - 2])),
        (0x0007, Val::Short(vec![width as u16 - 2])),
        (0x0009, Val::Short(vec![1])),
        (0x000A, Val::Short(vec![bits as u16])),
        (0x000E, Val::Short(vec![white])),
        (0x000F, Val::Short(vec![white])),
        (0x0010, Val::Short(vec![white])),
        (0x001C, Val::Short(vec![128])),
        (0x001D, Val::Short(vec![129])),
        (0x001E, Val::Short(vec![130])),
        (0x0024, Val::Short(vec![512])),
        (0x0025, Val::Short(vec![256])),
        (0x0026, Val::Short(vec![384])),
        (0x002D, Val::Short(vec![5])),
        (0x010F, Val::Ascii("Panasonic".into())),
        (0x0110, Val::Ascii("DC-Synthetic".into())),
        (0x0112, Val::Short(vec![1])),
        (0x0118, Val::Blobs(vec![raw])),
    ]);
    t.chain = vec![ifd0];
    let mut b = t.build();
    b[2..4].copy_from_slice(b"U\0");
    b
}

/// Byte split of the vendor-compressed (RawFormat 3/4) stream (see `rw2.rs`).
const RW2_COMPRESSED_SPLIT: usize = 0x2008;

/// Writer for the vendor-compressed RW2 stream: mirrors the decoder's cursor
/// (page addressing, downward field order) and deposits every field where the
/// reader will pick it up.
struct V4Writer<'a> {
    buf: &'a mut [u8],
    base: usize,
    pos: usize,
    nbits: u32,
}

impl V4Writer<'_> {
    /// Consumes one byte of the row seek (the decoder reads 8-bit fields).
    fn skip_byte(&mut self) {
        if 8 > self.nbits {
            self.nbits += (RW2_PAGE as u32) * 8;
            self.pos += RW2_PAGE;
        }
        self.nbits -= 8;
    }

    fn put(&mut self, v: u32, num: u32) {
        if num > self.nbits {
            self.nbits += (RW2_PAGE as u32) * 8;
            self.pos += RW2_PAGE;
        }
        let mut at = ((self.nbits - num) >> 3) as usize ^ 0x3FF0;
        at = (at + RW2_PAGE - RW2_COMPRESSED_SPLIT) % RW2_PAGE;
        let at = self.base + at + self.pos - RW2_PAGE;
        self.nbits -= num;
        let cur = u32::from(self.buf[at]) | (u32::from(self.buf[at + 1]) << 8);
        let cur = cur | ((v & ((1 << num) - 1)) << (self.nbits & 7));
        self.buf[at] = cur as u8;
        self.buf[at + 1] = (cur >> 8) as u8;
    }

    /// Encodes one row as 14-pixel groups: the scale selector is always 0 and
    /// the two first pixels of each group are absolute (8 + 4 bits); later
    /// pixels use the 8-bit delta path. Callers keep values inside 16..4096
    /// and stepping by at most ±127 between pixels of the same parity.
    fn row(&mut self, pixels: &[u16]) {
        for group in pixels.as_chunks::<14>().0 {
            let mut pred = [0i32; 2];
            let mut nonz = [0i32; 2];
            for (i, &px) in group.iter().enumerate() {
                if i % 3 == 2 {
                    self.put(0, 2); // sh = 0
                }
                let p = i & 1;
                let t = i32::from(px);
                if nonz[p] == 0 {
                    let nz = (t >> 4) & 0xFF;
                    self.put(nz as u32, 8);
                    if nz != 0 || i > 11 {
                        self.put((t & 0xF) as u32, 4);
                        pred[p] = t;
                    } else {
                        pred[p] = 0;
                    }
                    nonz[p] = nz;
                } else {
                    let q = (pred[p] - 0x80).max(0);
                    let j = t - q;
                    assert!((1..=0xFF).contains(&j), "{t} is not encodable from {}", pred[p]);
                    self.put(j as u32, 8);
                    pred[p] = q + j;
                }
                assert_eq!(pred[p], t);
            }
        }
    }
}

/// The vendor-compressed (RawFormat 3/4) stream of a whole image; see
/// [`V4Writer::row`] for the value constraints.
pub fn rw2_v4_pack(data: &[u16], width: usize) -> Vec<u8> {
    assert_eq!(width % 14, 0);
    let height = data.len() / width;
    assert_eq!(data.len(), width * height);
    let row_bytes = width / 14 * 16;
    let mut buf = vec![0u8; row_bytes * height + 2 * RW2_PAGE];
    for (row, pixels) in data.chunks_exact(width).enumerate() {
        let skip = row_bytes * row;
        let mut w = V4Writer { buf: &mut buf, base: skip / RW2_PAGE * RW2_PAGE, pos: 0, nbits: 0 };
        for _ in 0..(skip % RW2_PAGE) {
            w.skip_byte();
        }
        w.row(pixels);
    }
    buf
}

/// A synthetic Panasonic RW2 (RawFormat 4, vendor-compressed) with an RGGB
/// sensor, a 2-pixel border on every side, black levels 128 / 129 / 130 and
/// WB levels 512 / 256 / 384; see [`V4Writer::row`] for the pixel constraints.
pub fn rw2_format4(width: usize, height: usize, data: &[u16]) -> Vec<u8> {
    let mut t = TiffBuilder::default();
    let raw = t.blob(rw2_v4_pack(data, width));
    let ifd0 = t.ifd(vec![
        (0x0002, Val::Short(vec![width as u16])),
        (0x0003, Val::Short(vec![height as u16])),
        (0x0004, Val::Short(vec![2])),
        (0x0005, Val::Short(vec![2])),
        (0x0006, Val::Short(vec![height as u16 - 2])),
        (0x0007, Val::Short(vec![width as u16 - 2])),
        (0x0009, Val::Short(vec![1])),
        (0x000A, Val::Short(vec![12])),
        (0x000E, Val::Short(vec![4095])),
        (0x000F, Val::Short(vec![4095])),
        (0x0010, Val::Short(vec![4095])),
        (0x001C, Val::Short(vec![128])),
        (0x001D, Val::Short(vec![129])),
        (0x001E, Val::Short(vec![130])),
        (0x0024, Val::Short(vec![512])),
        (0x0025, Val::Short(vec![256])),
        (0x0026, Val::Short(vec![384])),
        (0x002D, Val::Short(vec![4])),
        (0x010F, Val::Ascii("Panasonic".into())),
        (0x0110, Val::Ascii("DC-Synthetic".into())),
        (0x0112, Val::Short(vec![1])),
        (0x0118, Val::Blobs(vec![raw])),
    ]);
    t.chain = vec![ifd0];
    let mut b = t.build();
    b[2..4].copy_from_slice(b"U\0");
    b
}

// ---------------------------------------------------------------- Olympus ORF

/// A little-endian IFD whose value offsets are relative to `base` (the IFD
/// itself sits at `base_offset` from that base). Entries: (tag, type, count,
/// value bytes).
fn relative_ifd(entries: &[(u16, u16, u32, Vec<u8>)], base_offset: usize) -> Vec<u8> {
    let head = 2 + 12 * entries.len() + 4;
    let mut out = Vec::new();
    let mut extra = Vec::new();
    out.extend_from_slice(&(entries.len() as u16).to_le_bytes());
    for (tag, typ, count, bytes) in entries {
        out.extend_from_slice(&tag.to_le_bytes());
        out.extend_from_slice(&typ.to_le_bytes());
        out.extend_from_slice(&count.to_le_bytes());
        if bytes.len() <= 4 {
            let mut v = bytes.clone();
            v.resize(4, 0);
            out.extend_from_slice(&v);
        } else {
            out.extend_from_slice(&((base_offset + head + extra.len()) as u32).to_le_bytes());
            extra.extend_from_slice(bytes);
        }
    }
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(&extra);
    out
}

fn shorts(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// A synthetic uncompressed Olympus ORF (`IIRO`): 12-bit `data` stored
/// left-justified in 16-bit samples, a GRBG EXIF CFAPattern and a new-style
/// maker note with ImageProcessing levels (black 64, WB 2.0 / 1.5, crop of a
/// 2-pixel border) and CameraSettings pointing at a tiny preview JPEG.
pub fn orf(width: usize, height: usize, data: &[u16]) -> Vec<u8> {
    let ip = relative_ifd(
        &[
            (0x0100, 3, 2, shorts(&[512, 384])),
            (0x011F, 3, 1, shorts(&[256])),
            (0x0600, 3, 4, shorts(&[64, 64, 64, 64])),
            (0x0611, 3, 2, shorts(&[12, 0])),
            (0x0612, 3, 2, shorts(&[2, 0])),
            (0x0613, 3, 2, shorts(&[2, 0])),
            (0x0614, 4, 1, ((width - 4) as u32).to_le_bytes().to_vec()),
            (0x0615, 4, 1, ((height - 4) as u32).to_le_bytes().to_vec()),
        ],
        64,
    );
    // CameraSettings with a preview JPEG (SOI, SOF0 16×8, EOI) right after it.
    let jpeg = [0xFF, 0xD8, 0xFF, 0xC0, 0x00, 0x0B, 8, 0x00, 0x08, 0x00, 0x10, 1, 1, 0x11, 0, 0xFF, 0xD9];
    let cs_at = 64 + ip.len().next_multiple_of(4);
    let cs_len = 2 + 12 * 3 + 4;
    let cs = relative_ifd(
        &[
            (0x0100, 4, 1, 1u32.to_le_bytes().to_vec()),
            (0x0101, 4, 1, ((cs_at + cs_len) as u32).to_le_bytes().to_vec()),
            (0x0102, 4, 1, (jpeg.len() as u32).to_le_bytes().to_vec()),
        ],
        cs_at,
    );
    let mut note = b"OLYMPUS\0II\x03\0".to_vec();
    note.extend_from_slice(&relative_ifd(&[(0x2020, 13, 1, (cs_at as u32).to_le_bytes().to_vec()), (0x2040, 13, 1, 64u32.to_le_bytes().to_vec())], 12));
    note.resize(64, 0);
    note.extend_from_slice(&ip);
    note.resize(cs_at, 0);
    note.extend_from_slice(&cs);
    note.extend_from_slice(&jpeg);
    let mut t = TiffBuilder::default();
    let strip = t.blob(data.iter().flat_map(|v| (v << 4).to_le_bytes()).collect());
    let exif = t.ifd(vec![(37500, Val::Undefined(note)), (41730, Val::Undefined(vec![2, 0, 2, 0, 1, 0, 2, 1]))]);
    let ifd0 = t.ifd(vec![
        (256, Val::Long(vec![width as u32])),
        (257, Val::Long(vec![height as u32])),
        (258, Val::Short(vec![16])),
        (259, Val::Short(vec![1])),
        (262, Val::Short(vec![1])),
        (271, Val::Ascii("OLYMPUS IMAGING CORP.".into())),
        (272, Val::Ascii("E-Synthetic".into())),
        (273, Val::Blobs(vec![strip])),
        (277, Val::Short(vec![1])),
        (278, Val::Long(vec![height as u32])),
        (279, Val::Long(vec![(data.len() * 2) as u32])),
        (34665, Val::Ifds(vec![exif])),
    ]);
    t.chain = vec![ifd0];
    let mut b = t.build();
    b[2..4].copy_from_slice(b"RO");
    b
}

// ---------------------------------------------------------------- Fujifilm RAF

/// How a synthetic RAF stores its samples (see `raf.rs`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RafPacking {
    /// 16-bit containers; little-endian (`II`) or big-endian (`MM`) CFA TIFF.
    U16 { le: bool },
    /// 12-bit, little-endian bit stream, least significant bit first.
    Lsb12,
    /// 14-bit, most significant bit first in little-endian 32-bit words.
    Words14,
    /// A short `IS…` strip standing in for Fujifilm's compressed data.
    Compressed,
}

/// The X-Trans repeat at the data origin, as observed (row major, 0 = R, 1 = G, 2 = B).
pub const XTRANS: [u8; 36] = [
    1, 1, 0, 1, 1, 2, //
    1, 1, 2, 1, 1, 0, //
    2, 0, 1, 0, 2, 1, //
    1, 1, 2, 1, 1, 0, //
    1, 1, 0, 1, 1, 2, //
    0, 2, 1, 2, 0, 1,
];

/// A synthetic Fujifilm RAF.
#[derive(Debug, Clone)]
pub struct RafSpec {
    pub width: usize,
    pub height: usize,
    pub data: Vec<u16>,
    pub bits: u32,
    pub packing: RafPacking,
    /// CFA colours at the data origin: 36 (X-Trans, 6×6) or 4 (Bayer, 2×2), row major.
    pub cfa: Vec<u8>,
    /// RawImageCropTopLeft / RawImageCroppedSize: top, left, height, width.
    pub crop: Option<[u16; 4]>,
    pub black: Vec<u32>,
    /// WB_GRBLevels: G, R, B.
    pub wb_grb: Option<[u32; 3]>,
    /// RawExposureBias as numerator / denominator.
    pub exposure_bias: Option<(i16, i16)>,
    /// EXIF orientation of the preview JPEG.
    pub orientation: u16,
    /// `false` writes an early-FinePix-style file without the CFA TIFF.
    pub cfa_tiff: bool,
}

impl RafSpec {
    pub fn new(width: usize, height: usize, data: Vec<u16>, cfa: Vec<u8>) -> Self {
        RafSpec {
            width,
            height,
            data,
            bits: 14,
            packing: RafPacking::U16 { le: true },
            cfa,
            crop: None,
            black: vec![1024],
            wb_grb: Some([302, 604, 453]),
            exposure_bias: None,
            orientation: 1,
            cfa_tiff: true,
        }
    }

    /// The samples as stored.
    fn strip(&self) -> Vec<u8> {
        match self.packing {
            RafPacking::U16 { le } => self.data.iter().flat_map(|v| if le { v.to_le_bytes() } else { v.to_be_bytes() }).collect(),
            RafPacking::Lsb12 => {
                let (mut out, mut acc, mut n) = (Vec::new(), 0u64, 0u32);
                for &v in &self.data {
                    acc |= u64::from(v & 0xFFF) << n;
                    n += 12;
                    while n >= 8 {
                        out.push(acc as u8);
                        acc >>= 8;
                        n -= 8;
                    }
                }
                if n > 0 {
                    out.push(acc as u8);
                }
                out
            }
            RafPacking::Words14 => {
                let (mut logical, mut acc, mut n) = (Vec::new(), 0u64, 0u32);
                for &v in &self.data {
                    acc = (acc << 14) | u64::from(v & 0x3FFF);
                    n += 14;
                    while n >= 8 {
                        logical.push((acc >> (n - 8)) as u8);
                        n -= 8;
                    }
                }
                if n > 0 {
                    logical.push((acc << (8 - n)) as u8);
                }
                logical.resize(logical.len().next_multiple_of(4), 0);
                logical.chunks(4).flat_map(|w| w.iter().rev().copied().collect::<Vec<u8>>()).collect()
            }
            RafPacking::Compressed => {
                let mut v = b"IS\x01\x00".to_vec();
                v.resize(self.width * self.height / 2, 0x5A);
                v
            }
        }
    }

    /// The CFA region: a TIFF whose IFD0 points (0xF000) at the Fuji raw IFD.
    fn cfa_region(&self) -> Vec<u8> {
        let le = !matches!(self.packing, RafPacking::U16 { le: false });
        let u16b = |v: u16| if le { v.to_le_bytes() } else { v.to_be_bytes() };
        let u32b = |v: u32| if le { v.to_le_bytes() } else { v.to_be_bytes() };
        let strip = self.strip();
        let strip_at = 2048u32;
        let entries: Vec<(u16, Vec<u32>)> = vec![
            (0xF001, vec![self.width as u32]),
            (0xF002, vec![self.height as u32]),
            (0xF003, vec![self.bits]),
            (0xF007, vec![strip_at]),
            (0xF008, vec![strip.len() as u32]),
            (0xF00A, self.black.clone()),
        ]
        .into_iter()
        .chain(self.wb_grb.map(|w| (0xF00E, w.to_vec())))
        .collect();
        let mut out = if le { b"II*\0".to_vec() } else { b"MM\0*".to_vec() };
        out.extend_from_slice(&u32b(8));
        // IFD0: one entry, 0xF000 (type IFD) → the Fuji IFD at 26.
        out.extend_from_slice(&u16b(1));
        out.extend_from_slice(&u16b(0xF000));
        out.extend_from_slice(&u16b(13));
        out.extend_from_slice(&u32b(1));
        out.extend_from_slice(&u32b(26));
        out.extend_from_slice(&u32b(0));
        let fuji_at = out.len();
        let mut extra_at = fuji_at + 2 + 12 * entries.len() + 4;
        let mut extra = Vec::new();
        out.extend_from_slice(&u16b(entries.len() as u16));
        for (tag, vals) in &entries {
            out.extend_from_slice(&u16b(*tag));
            out.extend_from_slice(&u16b(4));
            out.extend_from_slice(&u32b(vals.len() as u32));
            if vals.len() == 1 {
                out.extend_from_slice(&u32b(vals[0]));
            } else {
                out.extend_from_slice(&u32b(extra_at as u32));
                for v in vals {
                    extra.extend_from_slice(&u32b(*v));
                }
                extra_at += 4 * vals.len();
            }
        }
        out.extend_from_slice(&u32b(0));
        out.extend_from_slice(&extra);
        assert!(out.len() <= strip_at as usize);
        out.resize(strip_at as usize, 0);
        out.extend_from_slice(&strip);
        out
    }

    /// A tiny baseline-JPEG preview (APP1 EXIF with make, model and
    /// orientation; SOF0 32×24; EOI).
    fn preview(&self) -> Vec<u8> {
        let mut t = TiffBuilder::default();
        let ifd0 = t.ifd(vec![(271, Val::Ascii("FUJIFILM".into())), (272, Val::Ascii("X-Synthetic".into())), (274, Val::Short(vec![self.orientation]))]);
        t.chain = vec![ifd0];
        let mut app1 = b"Exif\0\0".to_vec();
        app1.extend_from_slice(&t.build());
        let mut j = vec![0xFF, 0xD8, 0xFF, 0xE1];
        j.extend_from_slice(&((app1.len() + 2) as u16).to_be_bytes());
        j.extend_from_slice(&app1);
        j.extend_from_slice(&[0xFF, 0xC0, 0x00, 0x0B, 8, 0x00, 24, 0x00, 32, 1, 1, 0x11, 0, 0xFF, 0xD9]);
        j
    }

    /// The metadata directory.
    fn meta(&self) -> Vec<u8> {
        let mut recs: Vec<(u16, Vec<u8>)> = Vec::new();
        if let Some([top, left, h, w]) = self.crop {
            recs.push((0x0110, [top.to_be_bytes(), left.to_be_bytes()].concat()));
            recs.push((0x0111, [h.to_be_bytes(), w.to_be_bytes()].concat()));
        }
        if self.cfa.len() == 36 {
            recs.push((0x0130, vec![0x0C; 4]));
            recs.push((0x0131, self.cfa.iter().rev().copied().collect()));
        } else {
            // Bayer: reversed, colour in the low two bits, the second green as 3.
            let mut greens = 0;
            let codes: Vec<u8> = self
                .cfa
                .iter()
                .map(|&c| {
                    if c == 1 {
                        greens += 1;
                        if greens == 1 { 0x09 } else { 0x0B }
                    } else {
                        0x08 | c
                    }
                })
                .collect();
            recs.push((0x0130, codes.into_iter().rev().collect()));
        }
        if let Some((n, d)) = self.exposure_bias {
            recs.push((0x9650, [n.to_be_bytes(), d.to_be_bytes()].concat()));
        }
        let mut out = (recs.len() as u32).to_be_bytes().to_vec();
        for (tag, d) in recs {
            out.extend_from_slice(&tag.to_be_bytes());
            out.extend_from_slice(&(d.len() as u16).to_be_bytes());
            out.extend_from_slice(&d);
        }
        out
    }

    pub fn build(&self) -> Vec<u8> {
        let mut b = b"FUJIFILMCCD-RAW 0201FF000000".to_vec();
        b.extend_from_slice(b"X-Synthetic");
        b.resize(60, 0);
        b.extend_from_slice(b"0100");
        b.resize(160, 0);
        let preview = self.preview();
        let meta = self.meta();
        let region = if self.cfa_tiff { self.cfa_region() } else { self.strip() };
        let (p_at, m_at) = (b.len(), b.len() + preview.len());
        let c_at = m_at + meta.len();
        for (at, v) in [(84, p_at), (88, preview.len()), (92, m_at), (96, meta.len()), (100, c_at), (104, region.len())] {
            b[at..at + 4].copy_from_slice(&(v as u32).to_be_bytes());
        }
        b.extend_from_slice(&preview);
        b.extend_from_slice(&meta);
        b.extend_from_slice(&region);
        b
    }
}

/// Mosaics linear RGB through any CFA repeat (`cfa`: `cfa_w` colours per
/// row) anchored at (0, 0), like [`mosaic`].
pub fn mosaic_cfa(rgb: &[[f32; 3]], width: usize, cfa: &[u8], cfa_w: usize, black: u16, white: u16) -> Vec<u16> {
    let cfa_h = cfa.len() / cfa_w;
    rgb.iter()
        .enumerate()
        .map(|(i, p)| {
            let (x, y) = (i % width, i / width);
            let c = cfa[(y % cfa_h) * cfa_w + x % cfa_w] as usize;
            (f32::from(black) + p[c].clamp(0.0, 1.0) * f32::from(white - black)).round() as u16
        })
        .collect()
}

// ---------------------------------------------------------------- scenes

/// A smooth, colourful synthetic scene as linear RGB in 0..1.
pub fn scene(width: usize, height: usize) -> Vec<[f32; 3]> {
    (0..width * height)
        .map(|i| {
            let (x, y) = ((i % width) as f32 / width.max(1) as f32, (i / width) as f32 / height.max(1) as f32);
            [0.1 + 0.7 * x, 0.15 + 0.5 * (1.0 - y) * x + 0.2 * y, 0.6 - 0.4 * y + 0.2 * x]
        })
        .collect()
}

/// Mosaics linear RGB through a 2×2 CFA (`cfa` row-major colours) into
/// sensor values `black + v * (white - black)`.
pub fn mosaic(rgb: &[[f32; 3]], width: usize, cfa: [u8; 4], black: u16, white: u16) -> Vec<u16> {
    rgb.iter()
        .enumerate()
        .map(|(i, p)| {
            let (x, y) = (i % width, i / width);
            let c = cfa[(y & 1) * 2 + (x & 1)] as usize;
            let v = p[c].clamp(0.0, 1.0);
            (f32::from(black) + v * f32::from(white - black)).round() as u16
        })
        .collect()
}
