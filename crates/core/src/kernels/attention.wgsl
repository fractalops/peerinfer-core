// Causal attention with grouped key-value heads over the cache.
// One workgroup of 256 threads per (head, token). D must be 64.

override HQ: u32;
override HKV: u32;
override D: u32;

const MAX_CTX: u32 = 2048u;
const SLICES: u32 = 4u; // 256 threads / 64 dims

struct Step {
    tokens: u32,
    pos0: u32,
}

@group(0) @binding(0) var<storage, read> qkv: array<f32>;
@group(0) @binding(1) var<storage, read> kc: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> vc: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> step: Step;

var<workgroup> sc: array<f32, MAX_CTX>;
var<workgroup> q: array<vec4<f32>, 16>;
var<workgroup> red: array<f32, 256>;

fn reduce(v: f32, i: u32, is_max: bool) -> f32 {
    red[i] = v;
    workgroupBarrier();
    for (var st = 128u; st > 0u; st >>= 1u) {
        if (i < st) {
            red[i] = select(red[i] + red[i + st], max(red[i], red[i + st]), is_max);
        }
        workgroupBarrier();
    }
    let r = red[0];
    workgroupBarrier();
    return r;
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) i: u32) {
    let h = wg.x;
    let t = wg.y;
    let len = step.pos0 + t + 1u;
    let kh = h / (HQ / HKV);
    let kvd = HKV * D;
    if (i < 16u) {
        let b = t * (HQ + 2u * HKV) * D + h * D + i * 4u;
        q[i] = vec4(qkv[b], qkv[b + 1u], qkv[b + 2u], qkv[b + 3u]);
    }
    workgroupBarrier();

    let inv = inverseSqrt(f32(D));
    var m = -3.4e38;
    for (var j = i; j < len; j += 256u) {
        let kb = (j * kvd + kh * D) / 4u;
        // Unrolled by hand (see matmul_tiled.wgsl).
        var d = 0.0;
        d += dot(q[0], kc[kb + 0u]) + dot(q[1], kc[kb + 1u]) + dot(q[2], kc[kb + 2u]) + dot(q[3], kc[kb + 3u]);
        d += dot(q[4], kc[kb + 4u]) + dot(q[5], kc[kb + 5u]) + dot(q[6], kc[kb + 6u]) + dot(q[7], kc[kb + 7u]);
        d += dot(q[8], kc[kb + 8u]) + dot(q[9], kc[kb + 9u]) + dot(q[10], kc[kb + 10u]) + dot(q[11], kc[kb + 11u]);
        d += dot(q[12], kc[kb + 12u]) + dot(q[13], kc[kb + 13u]) + dot(q[14], kc[kb + 14u]) + dot(q[15], kc[kb + 15u]);
        d *= inv;
        sc[j] = d;
        m = max(m, d);
    }
    m = reduce(m, i, true);

    var total = 0.0;
    for (var j = i; j < len; j += 256u) {
        let p = exp(sc[j] - m);
        sc[j] = p;
        total += p;
    }
    total = reduce(total, i, false);

    // Thread i sums dimension i % 64 over positions j = slice, slice + 4, ...
    let dim = i % D;
    let slice = i / D;
    // Four independent sums per thread; unrolled by hand.
    var acc = vec4<f32>();
    let vb = kh * D + dim;
    var j = slice;
    for (; j + 3u * SLICES < len; j += 4u * SLICES) {
        acc += vec4(
            sc[j] * vc[j * kvd + vb],
            sc[j + SLICES] * vc[(j + SLICES) * kvd + vb],
            sc[j + 2u * SLICES] * vc[(j + 2u * SLICES) * kvd + vb],
            sc[j + 3u * SLICES] * vc[(j + 3u * SLICES) * kvd + vb],
        );
    }
    for (; j < len; j += SLICES) {
        acc.x += sc[j] * vc[j * kvd + vb];
    }
    red[i] = acc.x + acc.y + acc.z + acc.w;
    workgroupBarrier();
    if (slice == 0u) {
        let a = red[dim] + red[dim + 64u] + red[dim + 128u] + red[dim + 192u];
        out[t * HQ * D + h * D + dim] = a / total;
    }
}
