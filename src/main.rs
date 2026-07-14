#![allow(unused_imports, dead_code, unused_variables, unused)]
use stellar::algebra::ndmethods::basic_mult;
use stellar::algebra::ndmethods::create_identity_matrix;
use stellar::algebra::ndmethods::matrix_mult;
use stellar::decomposition::lower_upper::LuPivotDecompose;
use stellar::decomposition::lq::AutumnDecomp;
use stellar::decomposition::schur::real_schur;
use stellar::decomposition::sgivens::{
    apply_g_left, apply_g_right, apply_gt_left, apply_gt_right, implicit_givens_rotation,
};
use stellar::equality::approximate::approx_vector_tol_eq;
use stellar::random::generation::{
    generate_identity_vector, generate_random_matrix, generate_random_vector,
};
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
pub fn params(v: &mut [f32], w: &mut [f32]) -> f32 {
    let mut max_element = 0f32;
    for val in v.iter() {
        let v = val.abs();
        if v > max_element {
            max_element = v
        };
    }
    if max_element.abs() < TOLERANCE {
        w[0] = 1f32;
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
/// rapply_householder
///
/// applies the transformation directly starting here to apply
/// to columns 1..cols, simply index into the data and then
/// stride = cols
/// cols = cols - 1;
///
/// * r: rotation matrix data
/// * p: projection vector
/// * w: workspace vector
/// * rows: number of rows
/// * cols: number of cols
/// * stride: stride of the data
fn lapply_householder(
    r: &mut [f32],
    p: &mut [f32],
    w: &mut [f32],
    tau: f32,
    rows: usize,
    cols: usize,
    stride: usize,
) {
    debug_assert!(cols <= w.len());
    // debug_assert_eq!(rows, p.len());
    // (I - tuu')A;
    // A -= t*uu'A;
    // w := u'A;
    // R -= t*uw';
    println!("rows {rows:}");
    println!("cols {cols:}");
    println!("r {r:?}");
    println!("p {p:?}");
    let mut roffset = 0;
    for j in 0..cols {
        // let scalar = p[0];
        // scalar implicitly 1
        w[j] = r[j];
    }
    println!("w first {w:?}");
    for i in 1..rows {
        roffset += stride;
        let scalar = p[i];
        for j in 0..cols {
            println!("roffset {}, j {}", roffset, j);
            w[j] += scalar * r[roffset + j];
        }
    }
    for j in 0..cols {
        w[j] *= tau;
        r[j] -= w[j];
    }
    println!("w {w:?}");
    roffset = 0;
    for i in 1..rows {
        println!("hello?");
        roffset += stride;
        for j in 0..cols {
            r[roffset + j] -= p[i] * w[j];
        }
    }
    println!("r {r:?}");
}
/// apply_householder
///
/// applies the transformation directly starting here to apply
/// to columns 1..cols, simply index into the data and then
/// stride = cols
/// cols = cols - 1;
///
/// * r: rotation matrix data
/// * p: projection vector
/// * w: workspace vector
/// * rows: number of rows
/// * cols: number of cols
/// * stride: stride of the data
fn rapply_householder(
    r: &mut [f32],
    p: &mut [f32],
    w: &mut [f32],
    tau: f32,
    rows: usize,
    cols: usize,
    stride: usize,
) {
    debug_assert!(rows <= w.len());
    // debug_assert_eq!(cols, p.len(), "cols {cols:}, p: {p:?}");
    // A(I - tuu');
    // A - t*Auu';
    // w := Au;
    // R -= t*wu;
    // println!("rows {rows:}");
    // println!("cols {cols:}");
    // println!("r {r:?}");
    // println!("r {r:?}");
    let mut roffset = 0;
    for i in 0..rows {
        w[i] = r[roffset] * p[0];
        for k in 1..cols {
            // println!("k {k:}");
            w[i] += r[roffset + k] * p[k - 1];
        }
        w[i] *= tau;
        roffset += stride;
    }
    // println!("w {w:?}");
    roffset = 0;
    for i in 0..rows {
        r[roffset] -= w[i];
        for j in 1..cols {
            r[roffset + j] -= w[i] * p[j - 1];
        }
        roffset += stride;
    }
}
struct FrancisLq {
    // kernel: NdArray,
    // transform: NdArray,
}
impl FrancisLq {
    /// full_hessenberg
    /// * h: matrix to create the hessenberg
    /// * r: rotation matrix should be identity on coldstart
    /// * p: projection vector
    /// * w: workspace vector
    /// * rows: number of rows
    /// * cols: number of cols
    /// * stride: stride of the data
    pub fn full_hessenberg(
        h: &mut [f32],
        r: &mut [f32],
        p: &mut [f32],
        w: &mut [f32],
        rows: usize,
        cols: usize,
        stride: usize,
    ) {
        // stores tau
        let mut offset = 0;
        let mut active_range = rows;
        let mut split_range = cols;
        for o in 1..2 {
            active_range -= 1;
            split_range -= 1;
            let (slice, target) = h.split_at_mut(offset + stride);
            let slice = &mut slice[offset + o..offset + cols];
            let proj = &mut p[..split_range];
            let tau = params(slice, proj);
            offset += stride;
            println!("proj {proj:?}");
            if tau == 0f32 {
                continue;
            }
            println!("r {r:?}");
            lapply_householder(&mut r[offset..], proj, w, tau, active_range, cols, stride);
            // rapply_householder(&mut h[o..], proj, w, tau, rows, split_range, stride);
        }
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
fn check_hessen() {
    let (rows, cols) = (3, 3);
    let stride = 3;
    let mut h = generate_random_vector(rows * cols);
    let mut r = generate_identity_vector(rows, cols);
    let mut p = vec![0f32; cols];
    let mut w = vec![0f32; rows];
    let input = NdArray {
        dims: vec![rows, cols],
        data: h.clone(),
    };
    println!("before {input:?}");
    FrancisLq::full_hessenberg(&mut h, &mut r, &mut p, &mut w, rows, cols, stride);
    let kernel = NdArray {
        dims: vec![rows, cols],
        data: h.clone(),
    };
    println!("after {kernel:?}");
    let rotation = NdArray {
        dims: vec![rows, cols],
        data: r.clone(),
    };
    println!("rotation {rotation:?}");
    let ortho = matrix_mult(&rotation, &rotation.transpose());
    println!("ortho rr' {ortho:?}");
    let ortho = matrix_mult(&rotation.transpose(), &rotation);
    println!("ortho r'r {ortho:?}");
    let reconstruct = matrix_mult(&kernel, &rotation);
    println!("reconstruct {reconstruct:?}");
}
// fn check() -> NdArray {
//     let d = 4;
//     let a = generate_random_matrix(d, d);
//     println!("a {a:?}");
//     let mut kern = vec![0f32; d * d];
//     let mut workspace = vec![0f32; d];
//     println!("kern {kern:?}");
//     let lq = AutumnDecomp::new(a);
//     lq.ql_apply(&mut kern, &mut workspace);
//     let input = NdArray {
//         dims: vec![d, d],
//         data: kern.clone(),
//     };
//     println!("before {input:?}");
//     francis_iteration(&mut kern, d, d);
//     let output = NdArray {
//         dims: vec![d, d],
//         data: kern.clone(),
//     };
//     println!("after {output:?}");
//     output
// }

fn main() {
    check_hessen();
    // test_reconstruct();
    // test_orthogonal();
}
// pub fn full_hessenberg(
//     h: &mut [f32],
//     r: &mut [f32],
//     p: &mut [f32],
//     w: &mut [f32],
//     rows: usize,
//     cols: usize,
//     stride: usize,
// ) {
//     // stores tau
//     let mut offset = 0;
//     let mut active_range = rows;
//     let mut split_range = cols;
//     println!("r {r:?}");
//     for o in 1..2 {
//         active_range -= 1;
//         split_range -= 1;
//         let (slice, target) = h.split_at_mut(offset + stride);
//         let slice = &mut slice[offset + o..offset + cols];
//         let proj = &mut p[..split_range];
//         let tau = params(slice, proj);
//         offset += stride;
//         println!("proj {proj:?}");
//         if tau == 0f32 { continue; }
//         // lapply_householder(&mut r[o..], proj, w, tau, rows, split_range, cols);
//         lapply_householder(&mut r[offset..], proj, w, tau, active_range, cols, stride);
//         rapply_householder(&mut r[o..], proj, w, tau, rows, split_range, stride);
//         // // let proj_suffix = &proj[1..];
//         // let proj_suffix = proj;
//         // let mut woffset = o;
//         // for i in 0..active_range {
//         //     let mut wi = target[woffset];
//         //     {
//         //         let targ_suffix = &mut target[woffset..woffset + split_range];
//         //         println!("targ_suffix {targ_suffix:?}");
//         //         println!("proj_suffix {proj_suffix:?}");
//         //         for j in 0..split_range {
//         //             wi += targ_suffix[j] * proj_suffix[j];
//         //         }
//         //         wi *= tau;
//         //         for j in 0..split_range {
//         //             targ_suffix[j] -= wi * proj_suffix[j];
//         //         }
//         //     }
//         //     // top left for 1f32
//         //     target[woffset] -= wi;
//         //     woffset += stride;
//         println!("r {r:?}");
//     }
// }
