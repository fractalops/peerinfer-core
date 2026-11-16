// Causal attention with grouped key-value heads over the cache.
// One workgroup of 256 threads per (head, token). D is 64 or 128.

override HQ: u32;
override HKV: u32;
override D: u32;

const MAX_CTX: u32 = 2048u;

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
var<workgroup> q: array<vec4<f32>, 32>;
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
    if (i < D / 4u) {
        let b = t * (HQ + 2u * HKV) * D + h * D + i * 4u;
        q[i] = vec4(qkv[b], qkv[b + 1u], qkv[b + 2u], qkv[b + 3u]);
    }
    workgroupBarrier();

    let inv = inverseSqrt(f32(D));
    var m = -3.4e38;
    for (var j = i; j < len; j += 256u) {
        let kb = (j * kvd + kh * D) / 4u;
        // Four dot products per iteration, unrolled by hand (see matmul_tiled.wgsl).
        var d = 0.0;
        for (var e = 0u; e < D / 4u; e += 4u) {
            d += dot(q[e], kc[kb + e]) + dot(q[e + 1u], kc[kb + e + 1u])
                + dot(q[e + 2u], kc[kb + e + 2u]) + dot(q[e + 3u], kc[kb + e + 3u]);
        }
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

    // Thread i sums dimension i % D over positions j = slice, slice + slices, ...
    let slices = 256u / D;
    let dim = i % D;
    let slice = i / D;
    // Four independent sums per thread; unrolled by hand.
    var acc = vec4<f32>();
    let vb = kh * D + dim;
    var j = slice;
    for (; j + 3u * slices < len; j += 4u * slices) {
        acc += vec4(
            sc[j] * vc[j * kvd + vb],
            sc[j + slices] * vc[(j + slices) * kvd + vb],
            sc[j + 2u * slices] * vc[(j + 2u * slices) * kvd + vb],
            sc[j + 3u * slices] * vc[(j + 3u * slices) * kvd + vb],
        );
    }
    for (; j < len; j += slices) {
        acc.x += sc[j] * vc[j * kvd + vb];
    }
    red[i] = acc.x + acc.y + acc.z + acc.w;
    workgroupBarrier();
    if (slice == 0u) {
        var a = 0.0;
        for (var r = 0u; r < slices; r++) {
            a += red[dim + r * D];
        }
        out[t * HQ * D + h * D + dim] = a / total;
    }
}
