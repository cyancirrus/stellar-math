// #![allow(unused_imports, dead_code, unused_variables, unused)]
use stellar::algebra::ndmethods::basic_mult;
use stellar::algebra::ndmethods::create_identity_matrix;
use stellar::decomposition::lower_upper::LuPivotDecompose;
use stellar::decomposition::lq::AutumnDecomp;
use stellar::decomposition::schur::real_schur;
use stellar::decomposition::sgivens::{
    apply_g_left, apply_g_right, apply_gt_left, apply_gt_right, implicit_givens_rotation,
};
use stellar::equality::approximate::approx_vector_tol_eq;
use stellar::random::generation::generate_random_matrix;
use stellar::structure::ndarray::NdArray;
// use stellar::decomposition::lq::params;
const TOLERANCE: f32 = 1e-4;

/// params
/// takes in data forom a matrix slice
/// zeros the incoming data and creates the householder vec
///
/// if the vector is less than the tolerance the workspace vec
/// will return nonsense
///
/// * v: matrix slice data
/// * w: sized workspace vector
pub fn params(v: &mut [f32], w:&mut [f32]) -> f32 {
    let mut max_element = 0f32;
    for val in v.iter() {
        let v = val.abs();
        if v > max_element {
            max_element = v
        };
    }
    if max_element.abs() < TOLERANCE {
        return 0f32;
    }
    let mut magnitude_squared = 0f32;
    let inv_max_element = 1f32 / max_element;
    for (val, gbg) in v.iter_mut().zip(w.iter_mut()) {
        *val *= inv_max_element;
        magnitude_squared += *val * *val;
        *gbg = *val;
        *val = 0f32;
    }
    let g = -w[0].signum() * magnitude_squared.sqrt();
    let scale = w[0] + g;
    let inv_scale = 1f32 / scale;
    for val in w[1..].iter_mut() {
        *val *= inv_scale;
    }
    v[0] = -g * max_element;
    w[0] = 1f32;
    scale / g
}
/// apply_householder
/// * r: rotation matrix data
/// * p: projection vector
/// * w: workspace vector
/// * dim: number of rows
/// * stride: stride of the data
fn apply_householder(
    r: &mut [f32],
    p: &mut [f32],
    w: &mut [f32],
    tau:f32,
    dim: usize,
    stride: usize
) {
    // A(I - tuu');
    // A - t*Auu';
    // v := Au;
    // R -= t*vu;
    let mut roffset = 0;
    for i in 0..dim {
        w[i] = r[roffset] * p[0];
        for k in 1..dim {
            w[i] += r[roffset + k] * p[k];
        }
        w[i] *= tau;
        roffset += stride;
    }
    roffset = 0;
    for i in 0..dim {
        for j in 0..dim {
            r[roffset + j] -= w[i] * p[j];
        }
        roffset += stride;
    }
}
struct FrancisLq {
    // kernel: NdArray,
    // transform: NdArray,
}
impl FrancisLq {
    // pub fn thing(self, h:&mut [f32], r:&mut [f32], w:&mut [f32], dim:usize, stride:usize) -> Self {
    //     self.zero(&mut h[1..], r[1..]
    // }

    pub fn zero(h: &mut [f32], r: &mut [f32], w: &mut [f32], dim: usize, stride: usize) -> Self {
        let (rows, cols) = (dim, dim);
        let mut t = vec![0f32; rows];
        let mut active_range = rows;
        for p in 0..rows {
            active_range -= 1;
            let tau = &mut t[p];
            let offset = p * stride;
            let (projection, target) = h.split_at_mut(offset + stride);
            let projection = &mut projection[offset + p..offset + stride];
            *tau = params(projection);
            if *tau == 0f32 {
                let roffset = p * cols;
                // h[roffset + p + 1..roffset + cols].fill(0f32);
                // continue;
            } else {
                apply_householder(
            }
            let proj_suffix = &projection[1..];
            let split_range = proj_suffix.len();
            for i in 0..active_range {
                let roffset = i * stride;
                let mut wi = target[roffset + p];
                {
                    let mut targ_suffix = &mut target[roffset + p + 1..roffset + cols];
                    targ_suffix = &mut targ_suffix[..split_range];
                    for j in 0..split_range {
                        wi += targ_suffix[j] * proj_suffix[j];
                    }
                    wi *= *tau;
                    for j in 0..split_range {
                        targ_suffix[j] -= wi * proj_suffix[j];
                    }
                }
                target[roffset + p] -= wi;
            }
        }
        Self {}
    }
}

fn eigen(m00: f32, m01: f32, m10: f32, m11: f32) -> f32 {
    let d = (m00 - m11) / 2f32;
    m11 + d - d.signum() * (d * d + m10 * m10).sqrt()
}
// h:= hessenberg matrix
fn decomp(h: &mut [f32], mut range: usize, stride: usize) {
    let s = range * stride;
    let mut e1 = s.saturating_sub(2);
    let mut e2 = s.saturating_sub(stride + 3);
    let mut eig: f32;
    while range != 0 {
        if h[e1].abs() < TOLERANCE {
            range -= 1;
            e1 = e1.saturating_sub(stride + 1);
            e2 = e2.saturating_sub(stride + 1);
        } else if h[e2].abs() < TOLERANCE {
            // if e2 == 0 then we are hitting eigen which should be greater than tolerance
            range -= 2;
            e1 = e1.saturating_sub(2 * stride + 2);
            e2 = e2.saturating_sub(2 * stride + 2);
        } else {
            francis_iteration(h, range, stride);
        }
    }
}
fn francis_iteration(h: &mut [f32], range: usize, stride: usize) {
    let card = stride * range;
    let tl = card.saturating_sub(stride + 2);
    let bl = card.saturating_sub(2);
    let eig = eigen(h[tl + 1], h[tl], h[bl + 1], h[bl]);
    let (_, cosine, sine) = implicit_givens_rotation(h[0] - eig, h[stride]);
    apply_g_left(h, 0, 1, stride, range, cosine, sine);
    apply_gt_right(h, 0, 1, stride, range, cosine, sine);
    for k in 1..range {
        let r = k * stride;
        let (_, cosine, sine) = implicit_givens_rotation(h[r + k], h[r + k + stride]);
        apply_gt_left(&mut h[r..], k, k + 1, stride, range - k, cosine, sine);
        apply_g_right(&mut h[r..], k, k + 1, stride, range - k, cosine, sine);
        // apply_g_left(&mut h[r..], k, k + 1, stride, range - k, cosine, sine);
        // apply_gt_right(&mut h[r..], k, k + 1, stride, range - k, cosine, sine);
    }
}
fn full_decomp(h: &mut [f32], t: &mut [f32], mut range: usize, stride: usize) {
    let card = range;
    let s = range * stride;
    let mut e1 = s.saturating_sub(2);
    let mut e2 = s.saturating_sub(stride + 3);
    let mut eig: f32;
    while range != 0 {
        if h[e1].abs() < TOLERANCE {
            range -= 1;
            e1 = e1.saturating_sub(stride + 1);
            e2 = e2.saturating_sub(stride + 1);
        } else if h[e2].abs() < TOLERANCE {
            // if e2 == 0 then we are hitting eigen which should be greater than tolerance
            range -= 2;
            e1 = e1.saturating_sub(2 * stride + 2);
            e2 = e2.saturating_sub(2 * stride + 2);
        } else {
            full_francis_iteration(h, t, card, range, stride);
        }
    }
}
fn full_francis_iteration(h: &mut [f32], t: &mut [f32], card: usize, range: usize, stride: usize) {
    let card = stride * range;
    let tl = card.saturating_sub(stride + 2);
    let bl = card.saturating_sub(2);
    let eig = eigen(h[tl], h[tl + 1], h[bl], h[bl + 1]);
    let (_, cosine, sine) = implicit_givens_rotation(h[0] - eig, h[stride]);
    apply_g_left(h, 0, 1, stride, range, cosine, sine);
    apply_gt_right(h, 0, 1, stride, range, cosine, sine);
    apply_g_right(t, 0, 1, stride, card, cosine, sine);
    for k in 1..range {
        let r = k * stride;
        let (_, cosine, sine) = implicit_givens_rotation(h[r + k], h[r + k + stride]);
        apply_g_left(&mut h[r..], k, k + 1, stride, range - k, cosine, sine);
        apply_gt_right(&mut h[r..], k, k + 1, stride, range - k, cosine, sine);
        apply_g_right(&mut t[r..], k, k + 1, stride, card, cosine, sine);
    }
}

fn check() -> NdArray {
    let d = 4;
    let a = generate_random_matrix(d, d);
    println!("a {a:?}");
    let mut kern = vec![0f32; d * d];
    let mut workspace = vec![0f32; d];
    println!("kern {kern:?}");
    let lq = AutumnDecomp::new(a);
    lq.ql_apply(&mut kern, &mut workspace);
    let input = NdArray {
        dims: vec![d, d],
        data: kern.clone(),
    };
    println!("before {input:?}");
    francis_iteration(&mut kern, d, d);
    let output = NdArray {
        dims: vec![d, d],
        data: kern.clone(),
    };
    println!("after {output:?}");
    output
}

fn main() {
    check();
    // test_reconstruct();
    // test_orthogonal();
}
