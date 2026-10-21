//! The 4×4 transforms. The inverse DCT and inverse WHT are the bit-exact procedures of RFC 6386
//! §14.3 and §14.4 (the encoder reconstructs with them so its prediction matches the decoder's).
//! The forward transforms are their derived counterparts: the inverse DCT is twice the
//! orthonormal transform per dimension and then `>> 3`, so coefficients are twice the
//! orthonormal DCT of the residual; the inverse WHT is an unnormalized Hadamard pair `>> 3`, so
//! the Y2 input is half the Hadamard pair of the DC values.

/// `sqrt(2) * cos(pi / 8) - 1` and `sqrt(2) * sin(pi / 8)` in 16.16 fixed point (RFC 6386 §14.4).
const C1: i32 = 20091;
const S1: i32 = 35468;

/// Residual (16 values, row-major) → coefficients (row-major; the zig-zag is applied when coding).
pub fn fdct(input: &[i32; 16]) -> [i32; 16] {
    let mut tmp = [0i32; 16];
    // Columns first (so that the row pass mirrors the decoder's order of rounding steps).
    for i in 0..4 {
        let (x0, x1, x2, x3) = (input[i], input[4 + i], input[8 + i], input[12 + i]);
        let a = x0 + x3;
        let b = x1 + x2;
        let c = x1 - x2;
        let d = x0 - x3;
        tmp[i] = a + b;
        tmp[8 + i] = a - b;
        // X1 = d * sqrt2cos(pi/8) + c * sqrt2sin(pi/8); X3 = d * sqrt2sin(pi/8) - c * sqrt2cos(pi/8).
        tmp[4 + i] = ((d * (65536 + C1) + c * S1 + 32768) >> 16).clamp(-32768, 32767);
        tmp[12 + i] = ((d * S1 - c * (65536 + C1) + 32768) >> 16).clamp(-32768, 32767);
    }
    let mut out = [0i32; 16];
    for r in 0..4 {
        let (x0, x1, x2, x3) = (tmp[r * 4], tmp[r * 4 + 1], tmp[r * 4 + 2], tmp[r * 4 + 3]);
        let a = x0 + x3;
        let b = x1 + x2;
        let c = x1 - x2;
        let d = x0 - x3;
        // Two passes give 4× the orthonormal transform; the coefficients are 2× it.
        out[r * 4] = (a + b + 1) >> 1;
        out[r * 4 + 2] = (a - b + 1) >> 1;
        out[r * 4 + 1] = ((d * (65536 + C1) + c * S1 + 65536) >> 17).clamp(-32768, 32767);
        out[r * 4 + 3] = ((d * S1 - c * (65536 + C1) + 65536) >> 17).clamp(-32768, 32767);
    }
    out
}

/// Dequantized coefficients → residual, bit-exact with the decoder (RFC 6386 §14.4).
pub fn idct(input: &[i32; 16]) -> [i32; 16] {
    let mut tmp = [0i32; 16];
    for i in 0..4 {
        let (ip0, ip4, ip8, ip12) = (input[i], input[4 + i], input[8 + i], input[12 + i]);
        let a1 = ip0 + ip8;
        let b1 = ip0 - ip8;
        let temp1 = (ip4 * S1) >> 16;
        let temp2 = ip12 + ((ip12 * C1) >> 16);
        let c1 = temp1 - temp2;
        let temp1 = ip4 + ((ip4 * C1) >> 16);
        let temp2 = (ip12 * S1) >> 16;
        let d1 = temp1 + temp2;
        tmp[i] = a1 + d1;
        tmp[12 + i] = a1 - d1;
        tmp[4 + i] = b1 + c1;
        tmp[8 + i] = b1 - c1;
    }
    let mut out = [0i32; 16];
    for r in 0..4 {
        let (ip0, ip1, ip2, ip3) = (tmp[r * 4], tmp[r * 4 + 1], tmp[r * 4 + 2], tmp[r * 4 + 3]);
        let a1 = ip0 + ip2;
        let b1 = ip0 - ip2;
        let temp1 = (ip1 * S1) >> 16;
        let temp2 = ip3 + ((ip3 * C1) >> 16);
        let c1 = temp1 - temp2;
        let temp1 = ip1 + ((ip1 * C1) >> 16);
        let temp2 = (ip3 * S1) >> 16;
        let d1 = temp1 + temp2;
        out[r * 4] = (a1 + d1 + 4) >> 3;
        out[r * 4 + 3] = (a1 - d1 + 4) >> 3;
        out[r * 4 + 1] = (b1 + c1 + 4) >> 3;
        out[r * 4 + 2] = (b1 - c1 + 4) >> 3;
    }
    out
}

/// The 16 luma DC values (row-major by sub-block) → Y2 coefficients.
pub fn fwht(input: &[i32; 16]) -> [i32; 16] {
    let mut tmp = [0i32; 16];
    for i in 0..4 {
        let (x0, x1, x2, x3) = (input[i], input[4 + i], input[8 + i], input[12 + i]);
        let a1 = x0 + x3;
        let b1 = x1 + x2;
        let c1 = x1 - x2;
        let d1 = x0 - x3;
        tmp[i] = a1 + b1;
        tmp[4 + i] = c1 + d1;
        tmp[8 + i] = a1 - b1;
        tmp[12 + i] = d1 - c1;
    }
    let mut out = [0i32; 16];
    for r in 0..4 {
        let (x0, x1, x2, x3) = (tmp[r * 4], tmp[r * 4 + 1], tmp[r * 4 + 2], tmp[r * 4 + 3]);
        let a1 = x0 + x3;
        let b1 = x1 + x2;
        let c1 = x1 - x2;
        let d1 = x0 - x3;
        // Half the unnormalized pair, rounded to nearest (ties away from zero).
        let half = |v: i32| (v + v.signum()) / 2;
        out[r * 4] = half(a1 + b1);
        out[r * 4 + 1] = half(c1 + d1);
        out[r * 4 + 2] = half(a1 - b1);
        out[r * 4 + 3] = half(d1 - c1);
    }
    out
}

/// Dequantized Y2 coefficients → the 16 luma DC values, bit-exact with the decoder (RFC 6386 §14.3).
pub fn iwht(input: &[i32; 16]) -> [i32; 16] {
    let mut tmp = [0i32; 16];
    for i in 0..4 {
        let a1 = input[i] + input[12 + i];
        let b1 = input[4 + i] + input[8 + i];
        let c1 = input[4 + i] - input[8 + i];
        let d1 = input[i] - input[12 + i];
        tmp[i] = a1 + b1;
        tmp[4 + i] = c1 + d1;
        tmp[8 + i] = a1 - b1;
        tmp[12 + i] = d1 - c1;
    }
    let mut out = [0i32; 16];
    for r in 0..4 {
        let a1 = tmp[r * 4] + tmp[r * 4 + 3];
        let b1 = tmp[r * 4 + 1] + tmp[r * 4 + 2];
        let c1 = tmp[r * 4 + 1] - tmp[r * 4 + 2];
        let d1 = tmp[r * 4] - tmp[r * 4 + 3];
        out[r * 4] = (a1 + b1 + 3) >> 3;
        out[r * 4 + 1] = (c1 + d1 + 3) >> 3;
        out[r * 4 + 2] = (a1 - b1 + 3) >> 3;
        out[r * 4 + 3] = (d1 - c1 + 3) >> 3;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: &mut u64) -> i32 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*seed >> 33) % 511) as i32 - 255
    }

    #[test]
    fn dct_inverts_within_rounding() {
        let mut seed = 7;
        for _ in 0..2000 {
            let input: [i32; 16] = std::array::from_fn(|_| lcg(&mut seed));
            let back = idct(&fdct(&input));
            for (a, b) in input.iter().zip(&back) {
                assert!((a - b).abs() <= 1, "{input:?} -> {back:?}");
            }
        }
    }

    #[test]
    fn dct_of_a_constant_block_is_a_dc_only_block() {
        let c = fdct(&[37; 16]);
        assert_eq!(c[0], 37 * 8, "DC is 8× the mean (2× orthonormal, 4 × 4 samples)");
        assert!(c[1..].iter().all(|&v| v == 0));
        assert_eq!(idct(&c), [37; 16]);
    }

    #[test]
    fn wht_inverts_exactly_in_the_decoders_range() {
        let mut seed = 11;
        for _ in 0..2000 {
            let input: [i32; 16] = std::array::from_fn(|_| lcg(&mut seed) * 8);
            let back = iwht(&fwht(&input));
            for (a, b) in input.iter().zip(&back) {
                assert!((a - b).abs() <= 1, "{input:?} -> {back:?}");
            }
        }
        let c = fwht(&[80; 16]);
        assert_eq!(c[0], 80 * 8);
        assert!(c[1..].iter().all(|&v| v == 0));
        assert_eq!(iwht(&c), [80; 16]);
    }
}
