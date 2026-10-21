//! Counter-based deterministic randomness.
//!
//! Every random value is a pure hash of `(seed, index, stream)`, so a dab's jitter depends only on
//! the stroke seed and the dab's position in the stroke — never on how the points were chunked,
//! on thread scheduling or on earlier draws. That is what makes journaled strokes replay exactly.

/// SplitMix64 finaliser.
#[inline]
pub fn mix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Uniform in [0, 1).
#[inline]
pub fn rand01(seed: u64, index: u64, stream: u64) -> f32 {
    let h = mix64(seed ^ mix64(index.wrapping_mul(0xD1B5_4A32_D192_ED03) ^ stream.wrapping_mul(0x8CB9_2BA7_2F3D_8DD7)));
    (h >> 40) as f32 / (1u64 << 24) as f32
}

/// Uniform in [-1, 1).
#[inline]
pub fn rand_signed(seed: u64, index: u64, stream: u64) -> f32 {
    rand01(seed, index, stream) * 2.0 - 1.0
}

/// Hash of integer document coordinates to [0, 1) (noise, dissolve).
#[inline]
pub fn hash2(x: i32, y: i32, salt: u64) -> f32 {
    let k = ((x as u32 as u64) << 32) | (y as u32 as u64);
    (mix64(k ^ salt.wrapping_mul(0xA24B_AED4_963E_E407)) >> 40) as f32 / (1u64 << 24) as f32
}

/// Streams: one per random quantity so they are independent.
pub mod stream {
    pub const SIZE: u64 = 1;
    pub const ANGLE: u64 = 2;
    pub const ROUNDNESS: u64 = 3;
    pub const FLIP_X: u64 = 4;
    pub const FLIP_Y: u64 = 5;
    pub const SCATTER_X: u64 = 6;
    pub const SCATTER_Y: u64 = 7;
    pub const COUNT: u64 = 8;
    pub const OPACITY: u64 = 9;
    pub const FLOW: u64 = 10;
    pub const FG_BG: u64 = 11;
    pub const HUE: u64 = 12;
    pub const SAT: u64 = 13;
    pub const BRIGHT: u64 = 14;
    pub const DEPTH: u64 = 15;
    pub const DUAL_X: u64 = 16;
    pub const DUAL_Y: u64 = 17;
    pub const DUAL_FLIP: u64 = 18;
    pub const WET: u64 = 19;
    pub const MIX: u64 = 20;
}

/// Seed derived from arbitrary bytes (FNV-1a then mixed), for commands without an explicit seed.
pub fn seed_from_bytes(b: &[u8]) -> u64 {
    let mut h: u64 = 0xCBF2_9CE4_8422_2325;
    for &x in b {
        h ^= u64::from(x);
        h = h.wrapping_mul(0x0000_0100_0000_01B3);
    }
    mix64(h)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_and_uniformish() {
        assert_eq!(rand01(7, 3, 1), rand01(7, 3, 1));
        assert_ne!(rand01(7, 3, 1), rand01(8, 3, 1));
        assert_ne!(rand01(7, 3, 1), rand01(7, 3, 2));
        let n = 10000;
        let mean: f32 = (0..n).map(|i| rand01(1, i, 0)).sum::<f32>() / n as f32;
        assert!((mean - 0.5).abs() < 0.02, "{mean}");
        assert!((0..n).all(|i| (0.0..1.0).contains(&rand01(2, i, 5))));
    }
}
