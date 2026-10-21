//! Minimal EXIF reader for the camera settings the photography commands use (exposure, aperture,
//! ISO, focal length, camera and lens names). Input is the raw TIFF-structured EXIF payload
//! (`II*\0` / `MM\0*`, as stored in `Document.metadata.exif`); malformed data yields `None`
//! fields, never a panic. Tag numbers are from the EXIF 2.32 specification (CIPA DC-008).

/// Camera settings read from EXIF.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CameraInfo {
    pub make: Option<String>,
    pub model: Option<String>,
    pub lens: Option<String>,
    /// Seconds.
    pub exposure_time: Option<f64>,
    pub f_number: Option<f64>,
    pub iso: Option<f64>,
    /// Millimetres.
    pub focal_length: Option<f64>,
    /// 35 mm-equivalent focal length in millimetres.
    pub focal_length_35mm: Option<f64>,
    /// DateTimeOriginal as stored (`YYYY:MM:DD HH:MM:SS`).
    pub date_taken: Option<String>,
}

impl CameraInfo {
    /// Linear exposure factor `t · ISO / N²` (relative brightness of the image for one scene).
    pub fn exposure_factor(&self) -> Option<f64> {
        let t = self.exposure_time?;
        let n = self.f_number.unwrap_or(1.0).max(0.1);
        Some(t * self.iso.unwrap_or(100.0) / 100.0 / (n * n))
    }
}

struct Tiff<'a> {
    b: &'a [u8],
    le: bool,
}

impl Tiff<'_> {
    fn u16(&self, o: usize) -> Option<u16> {
        let s = self.b.get(o..o.checked_add(2)?)?;
        Some(if self.le { u16::from_le_bytes([s[0], s[1]]) } else { u16::from_be_bytes([s[0], s[1]]) })
    }
    fn u32(&self, o: usize) -> Option<u32> {
        let s = self.b.get(o..o.checked_add(4)?)?;
        let a = [s[0], s[1], s[2], s[3]];
        Some(if self.le { u32::from_le_bytes(a) } else { u32::from_be_bytes(a) })
    }
    /// Entries of the IFD at `off`: (tag, type, count, value offset field position).
    fn ifd(&self, off: usize) -> Vec<(u16, u16, u32, usize)> {
        let Some(n) = self.u16(off) else { return Vec::new() };
        (0..n as usize)
            .filter_map(|k| {
                // Checked: offsets come from the file, and usize is 32 bits on wasm.
                let e = off.checked_add(2)?.checked_add(k * 12)?;
                Some((self.u16(e)?, self.u16(e.checked_add(2)?)?, self.u32(e.checked_add(4)?)?, e.checked_add(8)?))
            })
            .collect()
    }
    fn data_at(&self, typ: u16, count: u32, field: usize) -> Option<usize> {
        let size = match typ {
            1 | 2 | 6 | 7 => 1usize,
            3 | 8 => 2,
            4 | 9 | 11 => 4,
            5 | 10 | 12 => 8,
            _ => return None,
        }
        .checked_mul(usize::try_from(count).ok()?)?;
        let at = if size <= 4 { field } else { usize::try_from(self.u32(field)?).ok()? };
        // The whole value must lie inside the payload, not just its first element.
        self.b.get(at..at.checked_add(size)?)?;
        Some(at)
    }
    fn number(&self, typ: u16, count: u32, field: usize) -> Option<f64> {
        let at = self.data_at(typ, count, field)?;
        match typ {
            3 => self.u16(at).map(f64::from),
            4 => self.u32(at).map(f64::from),
            5 => {
                let (n, d) = (self.u32(at)?, self.u32(at.checked_add(4)?)?);
                (d != 0).then(|| n as f64 / d as f64)
            }
            10 => {
                let (n, d) = (self.u32(at)? as i32, self.u32(at.checked_add(4)?)? as i32);
                (d != 0).then(|| n as f64 / d as f64)
            }
            _ => None,
        }
    }
    fn text(&self, typ: u16, count: u32, field: usize) -> Option<String> {
        if typ != 2 {
            return None;
        }
        let at = self.data_at(typ, count, field)?;
        let s = self.b.get(at..at.checked_add(usize::try_from(count).ok()?)?)?;
        let s = String::from_utf8_lossy(s).trim_end_matches('\0').trim().to_string();
        (!s.is_empty()).then_some(s)
    }
}

/// Reads [`CameraInfo`] from a TIFF-structured EXIF payload (an `Exif\0\0` prefix is skipped).
pub fn read(exif: &[u8]) -> CameraInfo {
    let b = exif.strip_prefix(b"Exif\0\0").unwrap_or(exif);
    let mut info = CameraInfo::default();
    let le = match b.get(..4) {
        Some([b'I', b'I', 42, 0]) => true,
        Some([b'M', b'M', 0, 42]) => false,
        _ => return info,
    };
    let t = Tiff { b, le };
    let Some(ifd0) = t.u32(4) else { return info };
    let mut queue = vec![ifd0 as usize];
    let mut seen = Vec::new();
    while let Some(off) = queue.pop() {
        if seen.contains(&off) || seen.len() > 8 {
            continue;
        }
        seen.push(off);
        for (tag, typ, count, field) in t.ifd(off) {
            match tag {
                0x010F => info.make = t.text(typ, count, field),
                0x0110 => info.model = t.text(typ, count, field),
                0x829A => info.exposure_time = t.number(typ, count, field),
                0x829D => info.f_number = t.number(typ, count, field),
                0x8827 => info.iso = t.number(typ, count, field),
                0x920A => info.focal_length = t.number(typ, count, field),
                0xA405 => info.focal_length_35mm = t.number(typ, count, field).filter(|v| *v > 0.0),
                0xA434 => info.lens = t.text(typ, count, field),
                0x9003 => info.date_taken = t.text(typ, count, field),
                0x8769 => {
                    if let Some(p) = t.u32(field) {
                        queue.push(p as usize);
                    }
                }
                _ => {}
            }
        }
    }
    info
}

/// Builds a little-endian EXIF payload (used by tests and by synthetic fixtures).
pub fn build(info: &CameraInfo) -> Vec<u8> {
    // IFD0: Make, Model, ExifIFD pointer; Exif IFD: the rest. Data area after the IFDs.
    let mut ifd0: Vec<(u16, u16, u32, Vec<u8>)> = Vec::new();
    let mut exif: Vec<(u16, u16, u32, Vec<u8>)> = Vec::new();
    let text = |s: &str| {
        let mut v = s.as_bytes().to_vec();
        v.push(0);
        v
    };
    let rational = |v: f64| {
        let d = 10000u32;
        let mut b = ((v * d as f64).round() as u32).to_le_bytes().to_vec();
        b.extend_from_slice(&d.to_le_bytes());
        b
    };
    if let Some(s) = &info.make {
        ifd0.push((0x010F, 2, s.len() as u32 + 1, text(s)));
    }
    if let Some(s) = &info.model {
        ifd0.push((0x0110, 2, s.len() as u32 + 1, text(s)));
    }
    if let Some(v) = info.exposure_time {
        exif.push((0x829A, 5, 1, rational(v)));
    }
    if let Some(v) = info.f_number {
        exif.push((0x829D, 5, 1, rational(v)));
    }
    if let Some(v) = info.iso {
        exif.push((0x8827, 3, 1, (v as u16).to_le_bytes().to_vec()));
    }
    if let Some(s) = &info.date_taken {
        exif.push((0x9003, 2, s.len() as u32 + 1, text(s)));
    }
    if let Some(v) = info.focal_length {
        exif.push((0x920A, 5, 1, rational(v)));
    }
    if let Some(v) = info.focal_length_35mm {
        exif.push((0xA405, 3, 1, (v as u16).to_le_bytes().to_vec()));
    }
    if let Some(s) = &info.lens {
        exif.push((0xA434, 2, s.len() as u32 + 1, text(s)));
    }
    let ifd_len = |n: usize| 2 + 12 * n + 4;
    let ifd0_off = 8usize;
    let n0 = ifd0.len() + 1;
    let exif_off = ifd0_off + ifd_len(n0);
    let mut data_off = exif_off + ifd_len(exif.len());
    let mut out = b"II*\0".to_vec();
    out.extend_from_slice(&(ifd0_off as u32).to_le_bytes());
    let mut data = Vec::new();
    let mut write_ifd = |out: &mut Vec<u8>, entries: &[(u16, u16, u32, Vec<u8>)], extra: Option<u32>| {
        let n = entries.len() + usize::from(extra.is_some());
        out.extend_from_slice(&(n as u16).to_le_bytes());
        for (tag, typ, count, bytes) in entries {
            out.extend_from_slice(&tag.to_le_bytes());
            out.extend_from_slice(&typ.to_le_bytes());
            out.extend_from_slice(&count.to_le_bytes());
            if bytes.len() <= 4 {
                let mut f = bytes.clone();
                f.resize(4, 0);
                out.extend_from_slice(&f);
            } else {
                out.extend_from_slice(&(data_off as u32).to_le_bytes());
                data.extend_from_slice(bytes);
                data_off += bytes.len();
            }
        }
        if let Some(p) = extra {
            out.extend_from_slice(&0x8769u16.to_le_bytes());
            out.extend_from_slice(&4u16.to_le_bytes());
            out.extend_from_slice(&1u32.to_le_bytes());
            out.extend_from_slice(&p.to_le_bytes());
        }
        out.extend_from_slice(&0u32.to_le_bytes());
    };
    write_ifd(&mut out, &ifd0, Some(exif_off as u32));
    write_ifd(&mut out, &exif, None);
    out.extend_from_slice(&data);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_and_survives_garbage() {
        let info = CameraInfo {
            make: Some("Generic".into()),
            model: Some("Camera 1".into()),
            lens: Some("24-70mm".into()),
            exposure_time: Some(0.004),
            f_number: Some(8.0),
            iso: Some(200.0),
            focal_length: Some(35.0),
            focal_length_35mm: Some(52.0),
            date_taken: Some("2024:02:17 12:40:19".into()),
        };
        let b = build(&info);
        let r = read(&b);
        assert_eq!(r, info);
        assert!((r.exposure_factor().unwrap() - 0.004 * 2.0 / 64.0).abs() < 1e-9);
        let mut prefixed = b"Exif\0\0".to_vec();
        prefixed.extend_from_slice(&b);
        assert_eq!(read(&prefixed), info);
        assert_eq!(read(b"nonsense"), CameraInfo::default());
        for cut in [5, 12, 30, b.len() - 3] {
            let _ = read(&b[..cut]);
        }
    }

    /// Hostile payloads: huge counts, offsets near `u32::MAX`, self-referencing IFDs and random
    /// bytes read as `None` fields instead of panicking or overflowing (usize is 32 bits on wasm).
    #[test]
    fn hostile_payloads_never_panic() {
        let entry = |tag: u16, typ: u16, count: u32, value: u32| {
            let mut e = tag.to_le_bytes().to_vec();
            e.extend_from_slice(&typ.to_le_bytes());
            e.extend_from_slice(&count.to_le_bytes());
            e.extend_from_slice(&value.to_le_bytes());
            e
        };
        let mut b = b"II*\0".to_vec();
        b.extend_from_slice(&8u32.to_le_bytes());
        let entries = [
            entry(0x010F, 2, u32::MAX, 8),      // Make: count far past the end
            entry(0x0110, 2, 16, u32::MAX - 1), // Model: offset at the top of the range
            entry(0x829A, 5, u32::MAX, 20),     // ExposureTime: count overflows the size
            entry(0x829D, 5, 1, u32::MAX - 3),  // FNumber: rational straddles u32::MAX
            entry(0x9003, 2, 0x4000_0000, 8),   // DateTimeOriginal: size overflow on 32 bits
            entry(0x8769, 4, 1, 8),             // Exif IFD pointing back at IFD0
            entry(0x8769, 4, 1, u32::MAX),      // ... and past the end
        ];
        b.extend_from_slice(&(entries.len() as u16 + 3).to_le_bytes()); // claims more entries than exist
        for e in entries {
            b.extend_from_slice(&e);
        }
        assert_eq!(read(&b), CameraInfo::default());
        let mut be = b"MM\0*".to_vec();
        be.extend_from_slice(&u32::MAX.to_be_bytes());
        assert_eq!(read(&be), CameraInfo::default());
        // Deterministic noise behind a valid header, and every truncation of a real payload.
        let mut x = 0x2545_F491u32;
        for _ in 0..200 {
            let mut n = b"II*\0\x08\0\0\0".to_vec();
            for _ in 0..96 {
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                n.push(x as u8);
            }
            let _ = read(&n);
        }
        let good =
            build(&CameraInfo { make: Some("Make".into()), date_taken: Some("2024:02:17 12:40:19".into()), exposure_time: Some(0.004), ..Default::default() });
        for cut in 0..good.len() {
            let _ = read(&good[..cut]);
        }
    }
}
