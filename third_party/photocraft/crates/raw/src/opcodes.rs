//! DNG opcode lists (DNG 1.7, chapter 7 "Opcode List Processing").
//!
//! An opcode list is big-endian whatever the file's byte order: a count, then
//! per opcode its ID, DNG version, flags (bit 0 = optional), parameter size
//! and parameters. Only GainMap (ID 9), the lens-shading correction phones
//! write in OpcodeList2, is applied; other opcodes are reported.

/// One GainMap opcode: a grid of gains, bilinearly interpolated, applied to
/// the pixels of an area (in active-area coordinates) at a row / column pitch.
#[derive(Debug, Clone, PartialEq)]
pub struct GainMap {
    pub top: usize,
    pub left: usize,
    pub bottom: usize,
    pub right: usize,
    pub plane: usize,
    pub planes: usize,
    pub row_pitch: usize,
    pub col_pitch: usize,
    pub points_v: usize,
    pub points_h: usize,
    /// Spacing and origin of the map points, in image-relative coordinates (0..1).
    pub spacing_v: f64,
    pub spacing_h: f64,
    pub origin_v: f64,
    pub origin_h: f64,
    pub map_planes: usize,
    /// `points_v × points_h × map_planes` gains.
    pub gains: Vec<f32>,
}

impl GainMap {
    /// Gains along the map's horizontal points for image-relative row `rv`
    /// and map plane `mp` (vertical interpolation done).
    pub fn row(&self, rv: f64, mp: usize) -> Vec<f32> {
        let mp = mp.min(self.map_planes.saturating_sub(1));
        let fv = ((rv - self.origin_v) / self.spacing_v).clamp(0.0, (self.points_v - 1) as f64);
        let i0 = (fv.floor() as usize).min(self.points_v - 1);
        let i1 = (i0 + 1).min(self.points_v - 1);
        let t = (fv - i0 as f64) as f32;
        (0..self.points_h)
            .map(|j| {
                let a = self.gains.get((i0 * self.points_h + j) * self.map_planes + mp).copied().unwrap_or(1.0);
                let b = self.gains.get((i1 * self.points_h + j) * self.map_planes + mp).copied().unwrap_or(1.0);
                a + (b - a) * t
            })
            .collect()
    }

    /// Interpolates a [`GainMap::row`] at image-relative column `rh`.
    #[inline]
    pub fn at(&self, row: &[f32], rh: f64) -> f32 {
        let fh = ((rh - self.origin_h) / self.spacing_h).clamp(0.0, (self.points_h - 1) as f64);
        let j0 = (fh.floor() as usize).min(self.points_h - 1);
        let j1 = (j0 + 1).min(self.points_h - 1);
        let t = (fh - j0 as f64) as f32;
        let (a, b) = (row.get(j0).copied().unwrap_or(1.0), row.get(j1).copied().unwrap_or(1.0));
        a + (b - a) * t
    }

    /// `true` if the map covers active-area pixel (`ax`, `ay`).
    #[inline]
    pub fn covers(&self, ax: usize, ay: usize) -> bool {
        ay >= self.top
            && ay < self.bottom
            && ax >= self.left
            && ax < self.right
            && (ay - self.top).is_multiple_of(self.row_pitch)
            && (ax - self.left).is_multiple_of(self.col_pitch)
    }
}

/// Opcode names (DNG 1.7 table) for warnings.
fn name(id: u32) -> String {
    match id {
        1 => "WarpRectilinear".into(),
        2 => "WarpFisheye".into(),
        3 => "FixVignetteRadial".into(),
        4 => "FixBadPixelsConstant".into(),
        5 => "FixBadPixelsList".into(),
        6 => "TrimBounds".into(),
        7 => "MapTable".into(),
        8 => "MapPolynomial".into(),
        9 => "GainMap".into(),
        10 => "DeltaPerRow".into(),
        11 => "DeltaPerColumn".into(),
        12 => "ScalePerRow".into(),
        13 => "ScalePerColumn".into(),
        14 => "WarpRectilinear2".into(),
        id => format!("opcode {id}"),
    }
}

fn be32(b: &[u8], at: usize) -> Option<u32> {
    let s: [u8; 4] = b.get(at..at.checked_add(4)?)?.try_into().ok()?;
    Some(u32::from_be_bytes(s))
}

fn be64f(b: &[u8], at: usize) -> Option<f64> {
    let s: [u8; 8] = b.get(at..at.checked_add(8)?)?.try_into().ok()?;
    Some(f64::from_be_bytes(s)).filter(|v| v.is_finite())
}

fn gain_map(p: &[u8]) -> Option<GainMap> {
    let u = |i: usize| be32(p, 4 * i).map(|v| v as usize);
    let (top, left, bottom, right) = (u(0)?, u(1)?, u(2)?, u(3)?);
    let (plane, planes, row_pitch, col_pitch) = (u(4)?, u(5)?, u(6)?, u(7)?);
    let (points_v, points_h) = (u(8)?, u(9)?);
    let (spacing_v, spacing_h, origin_v, origin_h) = (be64f(p, 40)?, be64f(p, 48)?, be64f(p, 56)?, be64f(p, 64)?);
    let map_planes = u(18)?;
    let n = points_v.checked_mul(points_h)?.checked_mul(map_planes)?;
    if n == 0 || n > 1 << 20 || row_pitch == 0 || col_pitch == 0 || spacing_v <= 0.0 || spacing_h <= 0.0 || bottom <= top || right <= left {
        return None;
    }
    let gains = (0..n).map(|i| be32(p, 76 + 4 * i).map(f32::from_bits).filter(|g| g.is_finite())).collect::<Option<Vec<f32>>>()?;
    Some(GainMap {
        top,
        left,
        bottom,
        right,
        plane,
        planes,
        row_pitch,
        col_pitch,
        points_v,
        points_h,
        spacing_v,
        spacing_h,
        origin_v,
        origin_h,
        map_planes,
        gains,
    })
}

/// Parses an opcode list: the gain maps, and the names of opcodes not applied.
pub(crate) fn parse(list: &[u8]) -> (Vec<GainMap>, Vec<String>) {
    let mut maps = Vec::new();
    let mut skipped = Vec::new();
    let Some(count) = be32(list, 0) else { return (maps, skipped) };
    let mut at = 4usize;
    for _ in 0..count.min(4096) {
        let (Some(id), Some(size)) = (be32(list, at), be32(list, at.saturating_add(12))) else { break };
        let start = at.saturating_add(16);
        let Some(params) = list.get(start..start.saturating_add(size as usize)) else { break };
        match (id, gain_map(params)) {
            (9, Some(m)) => maps.push(m),
            _ => {
                let n = name(id);
                if !skipped.contains(&n) {
                    skipped.push(n);
                }
            }
        }
        at = at.saturating_add(16).saturating_add(size as usize);
    }
    (maps, skipped)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn list(ops: &[(u32, Vec<u8>)]) -> Vec<u8> {
        let mut b = (ops.len() as u32).to_be_bytes().to_vec();
        for (id, p) in ops {
            b.extend_from_slice(&id.to_be_bytes());
            b.extend_from_slice(&0x0103_0000u32.to_be_bytes());
            b.extend_from_slice(&1u32.to_be_bytes());
            b.extend_from_slice(&(p.len() as u32).to_be_bytes());
            b.extend_from_slice(p);
        }
        b
    }

    pub(crate) fn gain_map_params(area: [u32; 4], pitch: u32, gains: &[[f32; 2]; 2]) -> Vec<u8> {
        let mut p = Vec::new();
        for v in [area[0], area[1], area[2], area[3], 0, 1, pitch, pitch, 2, 2] {
            p.extend_from_slice(&v.to_be_bytes());
        }
        for v in [1.0f64, 1.0, 0.0, 0.0] {
            p.extend_from_slice(&v.to_be_bytes());
        }
        p.extend_from_slice(&1u32.to_be_bytes());
        for row in gains {
            for g in row {
                p.extend_from_slice(&g.to_bits().to_be_bytes());
            }
        }
        p
    }

    #[test]
    fn parses_gain_maps_and_names_others() {
        let b = list(&[(9, gain_map_params([0, 0, 10, 10], 1, &[[1.0, 2.0], [3.0, 4.0]])), (1, vec![0; 8])]);
        let (maps, skipped) = parse(&b);
        assert_eq!(maps.len(), 1);
        assert_eq!(skipped, vec!["WarpRectilinear".to_string()]);
        let m = &maps[0];
        // Bilinear: centre of the 2×2 grid.
        let row = m.row(0.5, 0);
        assert!((m.at(&row, 0.5) - 2.5).abs() < 1e-6);
        assert!((m.at(&m.row(0.0, 0), 0.0) - 1.0).abs() < 1e-6);
        assert!((m.at(&m.row(1.0, 0), 1.0) - 4.0).abs() < 1e-6);
        // Outside the grid clamps.
        assert!((m.at(&m.row(-3.0, 0), 9.0) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn hostile_lists_do_not_panic() {
        let b = list(&[(9, gain_map_params([0, 0, 10, 10], 2, &[[1.0, 2.0], [3.0, 4.0]]))]);
        for n in 0..b.len() {
            let _ = parse(&b[..n]);
        }
        let mut huge = b.clone();
        huge[4 + 16 + 32..4 + 16 + 40].copy_from_slice(&[0xFF; 8]); // points
        assert!(parse(&huge).0.is_empty());
    }
}
