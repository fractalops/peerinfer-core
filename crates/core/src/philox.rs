//! Philox4x32-10 counter-based random numbers (Salmon et al., "Parallel
//! random numbers: as easy as 1, 2, 3", SC 2011). The sampling kernel
//! computes the same function on the GPU.

const M0: u32 = 0xD251_1F53;
const M1: u32 = 0xCD9E_8D57;
const W0: u32 = 0x9E37_79B9;
const W1: u32 = 0xBB67_AE85;

fn mulhilo(a: u32, b: u32) -> (u32, u32) {
    let p = u64::from(a) * u64::from(b);
    ((p >> 32) as u32, p as u32)
}

/// Returns Philox4x32 with 10 rounds for `counter` and `key`.
pub fn philox4x32_10(counter: [u32; 4], key: [u32; 2]) -> [u32; 4] {
    let mut c = counter;
    let mut k = key;
    for _ in 0..10 {
        let (hi0, lo0) = mulhilo(M0, c[0]);
        let (hi1, lo1) = mulhilo(M1, c[2]);
        c = [hi1 ^ c[1] ^ k[0], lo1, hi0 ^ c[3] ^ k[1], lo0];
        k = [k[0].wrapping_add(W0), k[1].wrapping_add(W1)];
    }
    c
}

/// Uniform value in [0, 1) for one sampling step, as the GPU computes it.
pub fn uniform(seed: u64, position: u32) -> f32 {
    let r = philox4x32_10([position, 0, 0, 0], [seed as u32, (seed >> 32) as u32]);
    (r[0] >> 8) as f32 / 16_777_216.0
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Known-answer vectors from the Random123 distribution:
    /// https://github.com/DEShawResearch/random123/blob/main/tests/kat_vectors
    #[test]
    fn known_answers() {
        let cases = [
            (
                [0, 0, 0, 0],
                [0, 0],
                [0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8],
            ),
            (
                [0xffffffff; 4],
                [0xffffffff; 2],
                [0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd],
            ),
            (
                [0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344],
                [0xa4093822, 0x299f31d0],
                [0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1],
            ),
        ];
        for (counter, key, want) in cases {
            assert_eq!(philox4x32_10(counter, key), want);
        }
    }
}
