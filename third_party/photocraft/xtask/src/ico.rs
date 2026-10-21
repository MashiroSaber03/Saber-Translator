//! `cargo xtask ico <out.ico> <in.png>…`: pack PNG images into a Windows `.ico`.
//!
//! Vista and later read PNG-compressed icon entries, so each PNG is stored as is. The size comes
//! from the PNG's IHDR chunk; 256 px is written as 0, per the ICO format.

use std::path::Path;

/// Width and height from a PNG's IHDR chunk.
pub fn png_size(png: &[u8]) -> Result<(u32, u32), String> {
    const SIG: &[u8] = b"\x89PNG\r\n\x1a\n";
    if png.len() < 24 || &png[..8] != SIG || &png[12..16] != b"IHDR" {
        return Err("not a PNG file".into());
    }
    let w = u32::from_be_bytes([png[16], png[17], png[18], png[19]]);
    let h = u32::from_be_bytes([png[20], png[21], png[22], png[23]]);
    Ok((w, h))
}

/// An ICO file holding `pngs`, which must be square and at most 256 px.
pub fn pack(pngs: &[Vec<u8>]) -> Result<Vec<u8>, String> {
    if pngs.is_empty() || pngs.len() > u16::MAX as usize {
        return Err("need at least one PNG".into());
    }
    let mut out = Vec::new();
    out.extend_from_slice(&0u16.to_le_bytes()); // reserved
    out.extend_from_slice(&1u16.to_le_bytes()); // type: icon
    out.extend_from_slice(&(pngs.len() as u16).to_le_bytes());
    let mut offset = 6 + 16 * pngs.len();
    for png in pngs {
        let (w, h) = png_size(png)?;
        if w != h || w == 0 || w > 256 {
            return Err(format!("icon images must be square and at most 256 px, got {w}x{h}"));
        }
        let dim = if w == 256 { 0 } else { w as u8 };
        out.extend_from_slice(&[dim, dim, 0, 0]); // width, height, palette size, reserved
        out.extend_from_slice(&1u16.to_le_bytes()); // colour planes
        out.extend_from_slice(&32u16.to_le_bytes()); // bits per pixel
        out.extend_from_slice(&(png.len() as u32).to_le_bytes());
        out.extend_from_slice(&(offset as u32).to_le_bytes());
        offset += png.len();
    }
    for png in pngs {
        out.extend_from_slice(png);
    }
    Ok(out)
}

pub fn run(args: &[&str]) -> Result<(), String> {
    let [out, inputs @ ..] = args else {
        return Err("usage: cargo xtask ico <out.ico> <in.png>…".into());
    };
    let pngs = inputs.iter().map(|p| std::fs::read(p).map_err(|e| format!("read {p}: {e}"))).collect::<Result<Vec<_>, _>>()?;
    let ico = pack(&pngs)?;
    std::fs::write(Path::new(out), &ico).map_err(|e| format!("write {out}: {e}"))?;
    println!("{out}: {} images, {} bytes", pngs.len(), ico.len());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_png(size: u32) -> Vec<u8> {
        let mut v = b"\x89PNG\r\n\x1a\n\0\0\0\x0dIHDR".to_vec();
        v.extend_from_slice(&size.to_be_bytes());
        v.extend_from_slice(&size.to_be_bytes());
        v.extend_from_slice(&[8, 6, 0, 0, 0, 1, 2, 3, 4]);
        v
    }

    #[test]
    fn packs_directory_and_payloads() {
        let a = fake_png(16);
        let b = fake_png(256);
        let ico = pack(&[a.clone(), b.clone()]).unwrap();
        assert_eq!(&ico[..6], &[0, 0, 1, 0, 2, 0]);
        // Entry 0: 16 px at offset 6 + 32.
        assert_eq!(&ico[6..8], &[16, 16]);
        assert_eq!(u32::from_le_bytes(ico[14..18].try_into().unwrap()), a.len() as u32);
        assert_eq!(u32::from_le_bytes(ico[18..22].try_into().unwrap()), 38);
        // Entry 1: 256 px stored as 0, right after the first payload.
        assert_eq!(&ico[22..24], &[0, 0]);
        assert_eq!(u32::from_le_bytes(ico[34..38].try_into().unwrap()), 38 + a.len() as u32);
        assert_eq!(&ico[38..38 + a.len()], &a[..]);
        assert_eq!(&ico[38 + a.len()..], &b[..]);
    }

    #[test]
    fn rejects_bad_input() {
        assert!(pack(&[]).is_err());
        assert!(pack(&[fake_png(512)]).is_err());
        assert!(pack(&[b"GIF89a".to_vec()]).is_err());
        assert_eq!(png_size(&fake_png(48)).unwrap(), (48, 48));
    }

    #[test]
    fn committed_icon_is_valid() {
        let path = crate::root().join("assets/app-icon/photocraft.ico");
        let Ok(ico) = std::fs::read(&path) else { return };
        assert_eq!(&ico[..4], &[0, 0, 1, 0]);
        let n = u16::from_le_bytes([ico[4], ico[5]]) as usize;
        assert!(n >= 4, "expected several sizes in {}", path.display());
    }
}
