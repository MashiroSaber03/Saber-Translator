//! Tile-local memoization in rotated screen coordinates. Each cell's radius
//! is independent of the output pixel, including the five-tap edge clamping.
use photocraft_geom::Rect;

pub(super) struct Screen {
    x0: i32,
    y0: i32,
    width: usize,
    height: usize,
    radii: Vec<Option<f32>>,
}

impl Screen {
    pub(super) fn new(out: Rect, origin: (f32, f32), rotation: (f32, f32), cell: f32) -> Option<Self> {
        if out.is_empty() || !cell.is_finite() || cell < 1.0 {
            return None;
        }
        let (s, co) = rotation;
        let (mut lo_x, mut lo_y, mut hi_x, mut hi_y) = (f32::INFINITY, f32::INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);
        for y in [out.y0, out.y1.checked_sub(1)?] {
            for x in [out.x0, out.x1.checked_sub(1)?] {
                let (fx, fy) = (x as f32 + 0.5 - origin.0, y as f32 + 0.5 - origin.1);
                let (u, v) = ((fx * co + fy * s) / cell, (-fx * s + fy * co) / cell);
                if !u.is_finite() || !v.is_finite() || u.abs() > 4_000_000.0 || v.abs() > 4_000_000.0 {
                    return None;
                }
                lo_x = lo_x.min(u);
                lo_y = lo_y.min(v);
                hi_x = hi_x.max(u);
                hi_y = hi_y.max(v);
            }
        }
        // Include the neighbouring dots and a rounding margin. If a key ever
        // falls outside this box, radius() computes it directly.
        let (x0, y0) = (lo_x.floor() as i32 - 2, lo_y.floor() as i32 - 2);
        let width = usize::try_from(hi_x.floor() as i32 + 3 - x0).ok()?;
        let height = usize::try_from(hi_y.floor() as i32 + 3 - y0).ok()?;
        let len = width.checked_mul(height)?;
        // At most 2 MiB per channel, even for an explicit oversized tile.
        if len > 256 * 1024 {
            return None;
        }
        let mut radii = Vec::new();
        radii.try_reserve_exact(len).ok()?;
        radii.resize(len, None);
        Some(Self { x0, y0, width, height, radii })
    }

    pub(super) fn radius(&mut self, x: f32, y: f32, compute: impl FnOnce() -> f32) -> f32 {
        // Keys must represent exact integers: don't alias non-finite or large
        // coordinates onto cells whose floating-point arithmetic differs.
        let slot = if x.is_finite() && y.is_finite() && x.abs() <= 4_000_001.0 && y.abs() <= 4_000_001.0 {
            usize::try_from(x as i32 - self.x0)
                .ok()
                .zip(usize::try_from(y as i32 - self.y0).ok())
                .filter(|&(cx, cy)| cx < self.width && cy < self.height)
                .and_then(|(cx, cy)| cy.checked_mul(self.width)?.checked_add(cx))
                .and_then(|i| self.radii.get_mut(i))
        } else {
            None
        };
        match slot {
            Some(value) => *value.get_or_insert_with(compute),
            None => compute(),
        }
    }
}
