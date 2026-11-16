// x[t] = dequantized embedding row tokens[t]. One thread per packed word.

override K: u32;

@group(0) @binding(0) var<storage, read> tokens: array<u32>;
@group(0) @binding(1) var<storage, read> w: array<u32>;
@group(0) @binding(2) var<storage, read> s: array<u32>;
@group(0) @binding(3) var<storage, read_write> x: array<f32>;

fn scale(i: u32) -> f32 {
    let p = unpack2x16float(s[i >> 1u]);
    return select(p.x, p.y, (i & 1u) == 1u);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(workgroup_id) wg: vec3<u32>) {
    let words = K / 8u;
    let wi = gid.x;
    if (wi >= words) {
        return;
    }
    let t = wg.y;
    let row = tokens[t];
    let q = w[row * words + wi];
    let sc = scale(row * (K / 32u) + wi / 4u);
    for (var j = 0u; j < 8u; j++) {
        x[t * K + wi * 8u + j] = (f32((q >> (4u * j)) & 15u) - 7.0) * sc;
    }
}
