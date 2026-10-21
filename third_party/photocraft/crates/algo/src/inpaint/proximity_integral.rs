//! Summed-area table restricted to the nonzero bounds of a healing hole.

pub(super) struct HoleIntegral {
    x0: i32,
    y0: i32,
    w: usize,
    h: usize,
    data: Vec<u32>,
}

impl HoleIntegral {
    pub(super) fn new(stride: usize, hole: &[bool], (x0, y0, x1, y1): (usize, usize, usize, usize)) -> Option<Self> {
        let (w, h) = (x1.checked_sub(x0)?, y1.checked_sub(y0)?);
        let row_stride = w.checked_add(1)?;
        let len = row_stride.checked_mul(h.checked_add(1)?)?;
        let mut data = Vec::new();
        data.try_reserve_exact(len).ok()?;
        data.resize(len, 0u32);
        for (y, sy) in (y0..y1).enumerate() {
            let base = sy.checked_mul(stride)?;
            let row = hole.get(base.checked_add(x0)?..base.checked_add(x1)?)?;
            let mut count = 0u32;
            for (x, &set) in row.iter().enumerate() {
                count = count.checked_add(u32::from(set))?;
                let above = *data.get(y * row_stride + x + 1)?;
                *data.get_mut((y + 1) * row_stride + x + 1)? = above.checked_add(count)?;
            }
        }
        Some(Self { x0: i32::try_from(x0).ok()?, y0: i32::try_from(y0).ok()?, w, h, data })
    }

    pub(super) fn count(&self, x0: i32, y0: i32, x1: i32, y1: i32) -> u32 {
        super::window_count(
            &self.data,
            self.w,
            self.h,
            x0.saturating_sub(self.x0),
            y0.saturating_sub(self.y0),
            x1.saturating_sub(self.x0),
            y1.saturating_sub(self.y0),
        )
    }
}

#[cfg(test)]
mod tests;
