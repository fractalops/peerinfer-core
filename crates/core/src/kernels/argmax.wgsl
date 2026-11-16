// out[0] = index of the largest logit. Ties go to the lower index.

override V: u32;

@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<u32>;

var<workgroup> best_v: array<f32, 256>;
var<workgroup> best_i: array<u32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li: u32) {
    var bv = -3.4e38;
    var bi = 0u;
    for (var k = li; k < V; k += 256u) {
        let v = logits[k];
        if (v > bv) {
            bv = v;
            bi = k;
        }
    }
    best_v[li] = bv;
    best_i[li] = bi;
    workgroupBarrier();
    for (var st = 128u; st > 0u; st >>= 1u) {
        if (li < st) {
            let ov = best_v[li + st];
            let oi = best_i[li + st];
            if (ov > best_v[li] || (ov == best_v[li] && oi < best_i[li])) {
                best_v[li] = ov;
                best_i[li] = oi;
            }
        }
        workgroupBarrier();
    }
    if (li == 0u) {
        out[0] = best_i[0];
    }
}
