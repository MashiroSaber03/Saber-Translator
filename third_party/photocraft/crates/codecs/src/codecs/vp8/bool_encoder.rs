//! The VP8 boolean entropy coder, writing side (RFC 6386 §7): every symbol is a bool with an
//! 8-bit probability of being zero; literals and tree-coded values are built from bools.

use super::tables::Tree;

/// Writes bools at given probabilities into a byte buffer.
pub struct BoolEncoder {
    out: Vec<u8>,
    range: u32,
    bottom: u32,
    bit_count: i32,
}

impl Default for BoolEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl BoolEncoder {
    pub fn new() -> Self {
        BoolEncoder { out: Vec::new(), range: 255, bottom: 0, bit_count: 24 }
    }

    /// Propagates a carry into the bytes already written (never past the start: the coded
    /// value is below one).
    fn add_one_to_output(&mut self) {
        for b in self.out.iter_mut().rev() {
            if *b == 255 {
                *b = 0;
            } else {
                *b += 1;
                return;
            }
        }
    }

    /// Writes `bit` whose probability of being 0 is `prob / 256`.
    pub fn put(&mut self, prob: u8, bit: bool) {
        let split = 1 + (((self.range - 1) * u32::from(prob)) >> 8);
        if bit {
            self.bottom = self.bottom.wrapping_add(split);
            self.range -= split;
        } else {
            self.range = split;
        }
        while self.range < 128 {
            self.range <<= 1;
            if self.bottom & (1 << 31) != 0 {
                self.add_one_to_output();
            }
            self.bottom <<= 1;
            self.bit_count -= 1;
            if self.bit_count == 0 {
                self.out.push((self.bottom >> 24) as u8);
                self.bottom &= (1 << 24) - 1;
                self.bit_count = 8;
            }
        }
    }

    /// A bool at probability 1/2.
    pub fn put_flag(&mut self, bit: bool) {
        self.put(128, bit);
    }

    /// An `n`-bit unsigned literal, most significant bit first, each bit at probability 1/2.
    pub fn put_literal(&mut self, n: u32, value: u32) {
        for i in (0..n).rev() {
            self.put_flag((value >> i) & 1 != 0);
        }
    }

    /// A tree-coded value: the path of bools from the root to the leaf `value`, each node's bool
    /// at `probs[node / 2]`.
    pub fn put_tree(&mut self, tree: Tree, probs: &[u8], value: u8) {
        self.put_tree_from(tree, probs, value, 0);
    }

    /// [`Self::put_tree`] starting at node `start` (2 skips the first branch of the coefficient tree).
    pub fn put_tree_from(&mut self, tree: Tree, probs: &[u8], value: u8, start: usize) {
        // Find the path to the leaf by searching the tree; trees are tiny (≤ 22 entries).
        let mut path = Vec::with_capacity(8);
        if !Self::find(tree, start, value, &mut path) {
            return;
        }
        for (node, bit) in path {
            self.put(probs.get(node >> 1).copied().unwrap_or(128), bit);
        }
    }

    fn find(tree: Tree, node: usize, value: u8, path: &mut Vec<(usize, bool)>) -> bool {
        for bit in [false, true] {
            let i = node + usize::from(bit);
            let Some(&entry) = tree.get(i) else { continue };
            path.push((node, bit));
            if entry <= 0 {
                if (-entry) as u8 == value {
                    return true;
                }
            } else if Self::find(tree, entry as usize, value, path) {
                return true;
            }
            path.pop();
        }
        false
    }

    /// Finishes the partition and returns its bytes.
    pub fn finish(mut self) -> Vec<u8> {
        let mut c = self.bit_count;
        let mut v = self.bottom;
        if c <= 32 && v & (1u32 << (32 - c)) != 0 {
            self.add_one_to_output();
        }
        v <<= (c & 7) as u32;
        c >>= 3;
        while c > 0 {
            v <<= 8;
            c -= 1;
        }
        for _ in 0..4 {
            self.out.push((v >> 24) as u8);
            v <<= 8;
        }
        self.out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The decoder from RFC 6386 §7.3, for the round-trip tests.
    struct BoolDecoder<'a> {
        input: &'a [u8],
        pos: usize,
        range: u32,
        value: u32,
        bit_count: i32,
    }

    impl<'a> BoolDecoder<'a> {
        fn new(input: &'a [u8]) -> Self {
            let mut value = 0;
            for i in 0..2 {
                value = (value << 8) | u32::from(input.get(i).copied().unwrap_or(0));
            }
            BoolDecoder { input, pos: 2, range: 255, value, bit_count: 0 }
        }
        fn get(&mut self, prob: u8) -> bool {
            let split = 1 + (((self.range - 1) * u32::from(prob)) >> 8);
            let big_split = split << 8;
            let bit = if self.value >= big_split {
                self.range -= split;
                self.value -= big_split;
                true
            } else {
                self.range = split;
                false
            };
            while self.range < 128 {
                self.value <<= 1;
                self.range <<= 1;
                self.bit_count += 1;
                if self.bit_count == 8 {
                    self.bit_count = 0;
                    self.value |= u32::from(self.input.get(self.pos).copied().unwrap_or(0));
                    self.pos += 1;
                }
            }
            bit
        }
        fn literal(&mut self, n: u32) -> u32 {
            (0..n).fold(0, |v, _| (v << 1) | u32::from(self.get(128)))
        }
        fn tree(&mut self, tree: Tree, probs: &[u8], start: usize) -> u8 {
            let mut i = start;
            loop {
                let bit = self.get(probs[i >> 1]);
                let entry = tree[i + usize::from(bit)];
                if entry <= 0 {
                    return (-entry) as u8;
                }
                i = entry as usize;
            }
        }
    }

    #[test]
    fn bools_round_trip_at_every_probability() {
        let mut rng = 0x1234_5678_9abc_def0u64;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        let symbols: Vec<(u8, bool)> = (0..20_000)
            .map(|_| {
                let r = next();
                let prob = (r & 0xff) as u8;
                // Bits follow their probability, roughly, so both paths are exercised.
                ((prob.max(1)), (r >> 8) & 0xff >= u64::from(prob))
            })
            .collect();
        let mut e = BoolEncoder::new();
        for &(p, b) in &symbols {
            e.put(p, b);
        }
        let bytes = e.finish();
        let mut d = BoolDecoder::new(&bytes);
        for &(p, b) in &symbols {
            assert_eq!(d.get(p), b);
        }
    }

    #[test]
    fn literals_and_trees_round_trip() {
        use super::super::tables::*;
        let mut e = BoolEncoder::new();
        e.put_literal(7, 93);
        e.put_literal(19, 0x7_ffff);
        e.put_literal(1, 0);
        e.put_tree(KF_YMODE_TREE, &KF_YMODE_PROB, TM_PRED);
        e.put_tree(KF_YMODE_TREE, &KF_YMODE_PROB, B_PRED);
        e.put_tree(UV_MODE_TREE, &KF_UV_MODE_PROB, H_PRED);
        for m in 0..10u8 {
            e.put_tree(BMODE_TREE, &KF_BMODE_PROBS[3][7], m);
        }
        let probs = [200u8, 150, 100, 128, 128, 128, 128, 128, 128, 128, 128];
        for t in 0..12u8 {
            e.put_tree(COEFF_TREE, &probs, t);
        }
        // From node 2 (after a zero token) the end-of-block leaf is unreachable: tokens 0..=10.
        for t in 0..11u8 {
            e.put_tree_from(COEFF_TREE, &probs, t, 2);
        }
        let bytes = e.finish();
        let mut d = BoolDecoder::new(&bytes);
        assert_eq!(d.literal(7), 93);
        assert_eq!(d.literal(19), 0x7_ffff);
        assert_eq!(d.literal(1), 0);
        assert_eq!(d.tree(KF_YMODE_TREE, &KF_YMODE_PROB, 0), TM_PRED);
        assert_eq!(d.tree(KF_YMODE_TREE, &KF_YMODE_PROB, 0), B_PRED);
        assert_eq!(d.tree(UV_MODE_TREE, &KF_UV_MODE_PROB, 0), H_PRED);
        for m in 0..10u8 {
            assert_eq!(d.tree(BMODE_TREE, &KF_BMODE_PROBS[3][7], 0), m);
        }
        for t in 0..12u8 {
            assert_eq!(d.tree(COEFF_TREE, &probs, 0), t);
        }
        for t in 0..11u8 {
            assert_eq!(d.tree(COEFF_TREE, &probs, 2), t);
        }
    }

    #[test]
    fn carry_propagates_through_full_bytes() {
        // Many ones at a tiny probability push the interval up and force carries.
        let mut e = BoolEncoder::new();
        for _ in 0..3000 {
            e.put(1, true);
            e.put(255, false);
        }
        let bytes = e.finish();
        let mut d = BoolDecoder::new(&bytes);
        for _ in 0..3000 {
            assert!(d.get(1));
            assert!(!d.get(255));
        }
    }
}
