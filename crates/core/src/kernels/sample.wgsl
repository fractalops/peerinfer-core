// Samples the next token from the logits with temperature and top-p, in
// one workgroup. Writes the token to tokens[0] (the input of the next
// decode step) and to history at its position, like argmax.wgsl.
//
// 1. p_i = exp((l_i - max) / temperature), kept in `probs`.
// 2. Top-p without sorting: bisection finds the largest threshold t such
//    that the tokens with p_i >= t hold at least top_p of the mass.
// 3. The draw: u = Philox4x32-10(counter = position, key = seed), scaled
//    to the kept mass, then a prefix sum over the kept tokens. Thread i
//    owns a contiguous range of tokens.
// tokens[1] receives the raw random word so tests can check the RNG.

override V: u32;

const THREADS: u32 = 256u;
const BISECT_STEPS: u32 = 24u;

struct Step {
    tokens: u32,
    pos0: u32,
}

struct Params {
    temperature: f32,
    top_p: f32,
    seed_lo: u32,
    seed_hi: u32,
}

@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read_write> probs: array<f32>;
@group(0) @binding(2) var<storage, read_write> tokens: array<u32>;
@group(0) @binding(3) var<storage, read_write> history: array<u32>;
@group(0) @binding(4) var<uniform> step: Step;
@group(0) @binding(5) var<uniform> params: Params;

var<workgroup> red: array<f32, 256>;
var<workgroup> red_i: array<u32, 256>;
var<workgroup> chosen: u32;

fn reduce_sum(v: f32, li: u32) -> f32 {
    red[li] = v;
    workgroupBarrier();
    for (var st = THREADS / 2u; st > 0u; st >>= 1u) {
        if (li < st) {
            red[li] += red[li + st];
        }
        workgroupBarrier();
    }
    let r = red[0];
    workgroupBarrier();
    return r;
}

// Returns (hi, lo) of the 64-bit product a * b.
fn mulhilo(a: u32, b: u32) -> vec2<u32> {
    let a0 = a & 0xFFFFu;
    let a1 = a >> 16u;
    let b0 = b & 0xFFFFu;
    let b1 = b >> 16u;
    let p0 = a0 * b0;
    let p1 = a0 * b1;
    let p2 = a1 * b0;
    let p3 = a1 * b1;
    let mid = (p0 >> 16u) + (p1 & 0xFFFFu) + (p2 & 0xFFFFu);
    let hi = p3 + (p1 >> 16u) + (p2 >> 16u) + (mid >> 16u);
    return vec2(hi, a * b);
}

fn philox(counter: vec4<u32>, key: vec2<u32>) -> vec4<u32> {
    var c = counter;
    var k = key;
    for (var r = 0u; r < 10u; r++) {
        let p0 = mulhilo(0xD2511F53u, c.x);
        let p1 = mulhilo(0xCD9E8D57u, c.z);
        c = vec4(p1.x ^ c.y ^ k.x, p1.y, p0.x ^ c.w ^ k.y, p0.y);
        k += vec2(0x9E3779B9u, 0xBB67AE85u);
    }
    return c;
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li: u32) {
    let per = (V + THREADS - 1u) / THREADS;
    let start = min(li * per, V);
    let end = min(start + per, V);

    // Largest logit, ties to the lower index.
    var bv = -3.4e38;
    var bi = 0u;
    for (var k = start; k < end; k++) {
        let v = logits[k];
        if (v > bv) {
            bv = v;
            bi = k;
        }
    }
    red[li] = bv;
    red_i[li] = bi;
    workgroupBarrier();
    for (var st = THREADS / 2u; st > 0u; st >>= 1u) {
        if (li < st) {
            let ov = red[li + st];
            let oi = red_i[li + st];
            if (ov > red[li] || (ov == red[li] && oi < red_i[li])) {
                red[li] = ov;
                red_i[li] = oi;
            }
        }
        workgroupBarrier();
    }
    let m = red[0];
    let argmax = red_i[0];
    workgroupBarrier();

    let pos = step.pos0 + step.tokens;
    let r = philox(vec4(pos, 0u, 0u, 0u), vec2(params.seed_lo, params.seed_hi));
    let u01 = f32(r.x >> 8u) / 16777216.0;

    // p_i relative to the largest, so the largest is 1.
    let inv_t = 1.0 / params.temperature;
    var local = 0.0;
    for (var k = start; k < end; k++) {
        let p = exp((logits[k] - m) * inv_t);
        probs[k] = p;
        local += p;
    }
    let total = reduce_sum(local, li);

    // Bisection on the threshold. lo always keeps at least top_p of the mass.
    var lo = 0.0;
    if (params.top_p < 1.0) {
        var hi = 1.0;
        let goal = params.top_p * total;
        for (var it = 0u; it < BISECT_STEPS; it++) {
            let mid = 0.5 * (lo + hi);
            var mass = 0.0;
            for (var k = start; k < end; k++) {
                let p = probs[k];
                mass += select(0.0, p, p >= mid);
            }
            if (reduce_sum(mass, li) >= goal) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
    }

    // Exclusive prefix sum of the kept mass over the threads.
    var kept = 0.0;
    for (var k = start; k < end; k++) {
        let p = probs[k];
        kept += select(0.0, p, p >= lo);
    }
    red[li] = kept;
    workgroupBarrier();
    for (var st = 1u; st < THREADS; st <<= 1u) {
        // Index first: select() evaluates both sides.
        let v = select(0.0, red[select(li, li - st, li >= st)], li >= st);
        workgroupBarrier();
        red[li] += v;
        workgroupBarrier();
    }
    let inclusive = red[li];
    let kept_total = red[THREADS - 1u];
    if (li == 0u) {
        chosen = argmax;
    }
    workgroupBarrier();

    // The thread whose range holds u picks the token inside it.
    let u = u01 * kept_total;
    let before = inclusive - kept;
    if (kept > 0.0 && u >= before && u < inclusive) {
        var acc = before;
        var pick = argmax;
        var found = false;
        for (var k = start; k < end; k++) {
            let p = probs[k];
            if (p >= lo && !found) {
                acc += p;
                pick = k; // the last kept token, if rounding leaves u past acc
                found = u < acc;
            }
        }
        chosen = pick;
    }
    workgroupBarrier();
    if (li == 0u) {
        tokens[0] = chosen;
        tokens[1] = r.x;
        history[pos] = chosen;
    }
}
