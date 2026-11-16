// Picks the largest logit (ties go to the lower index). Writes it to
// tokens[0], the input of the next decode step, and to history at the
// position the new token takes.

override V: u32;

struct Step {
    tokens: u32,
    pos0: u32,
}

@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read_write> tokens: array<u32>;
@group(0) @binding(2) var<storage, read_write> history: array<u32>;
@group(0) @binding(3) var<uniform> step: Step;

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
        tokens[0] = best_i[0];
        history[step.pos0 + step.tokens] = best_i[0];
    }
}
