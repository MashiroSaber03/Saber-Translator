//! Source pixels of canonical grid links touching a free pixel, in raster order.
//! Merge the free pixels and their left/upper three predecessors. Each stream
//! is sorted; a five-way merge deduplicates in O(free pixels), without a dense bitmap.

pub(super) struct Starts<'a> {
    free: &'a [usize],
    w: usize,
    at: [usize; 5],
    peek: [Option<usize>; 5],
}

impl<'a> Starts<'a> {
    pub(super) fn new(free: &'a [usize], w: usize) -> Self {
        let mut this = Self { free, w, at: [0; 5], peek: [None; 5] };
        for k in 0..5 {
            this.advance(k)
        }
        this
    }

    fn advance(&mut self, k: usize) {
        self.peek[k] = None;
        while let Some(&i) = self.free.get(self.at[k]) {
            self.at[k] += 1;
            let x = i % self.w;
            let value = match k {
                0 => Some(i),
                1 => (x > 0).then(|| i - 1),
                2 => (i >= self.w).then(|| i - self.w),
                3 => (i >= self.w && x > 0).then(|| i - self.w - 1),
                _ => (i >= self.w && x + 1 < self.w).then(|| i - self.w + 1),
            };
            if value.is_some() {
                self.peek[k] = value;
                break;
            }
        }
    }
}

impl Iterator for Starts<'_> {
    type Item = usize;
    fn next(&mut self) -> Option<usize> {
        let first = self.peek.iter().flatten().copied().min()?;
        for k in 0..5 {
            if self.peek[k] == Some(first) {
                self.advance(k)
            }
        }
        Some(first)
    }
}

#[cfg(test)]
mod tests;
