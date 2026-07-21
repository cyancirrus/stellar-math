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
// const TOLERANCE: f32 = 1e-8;
// const EPSILON: f32 = 1e-12;
// const TOLERANCE: f32 = 1e-6;
// const EPSILON: f32 = 1e-8;
const TOLERANCE: f32 = 1e-6;
const EPSILON: f32 = 1e-8;

// NOTE: To self householder vecs getting inf's, well it's bc of like [w0, 0, 0] style vecs
// symmetric case done


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
    // TODO: a bug when the other numbers aren't valid
    debug_assert_eq!(v.len(), w.len());
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
    println!("inv_max_element {inv_max_element:?}");
    for (val, gbg) in v.iter_mut().zip(w.iter_mut()) {
        *val *= inv_max_element;
        magnitude_squared += *val * *val;
        println!("val {}", *val);
        *gbg = *val;
        *val = 0f32;
    }
    let g = -w[0].signum() * magnitude_squared.sqrt();
    let scale = w[0] + g;
    let inv_scale = 1f32 / scale;
    println!("inv_scale {inv_scale:?}");
    for val in w[1..].iter_mut() {
        *val *= inv_scale;
    }
    v[0] = -g * max_element;
    w[0] = 1f32;
    println!("returning tau {:?}, scale {scale:}, g {g:}", scale / g);
    scale / g
}
/// lapply_householder
///
/// applies the transformation directly starting here to apply
/// to columns 1..cols, simply index into the data and then
/// stride = cols
/// cols = cols - 1;
///
/// * h: matrix linear data slice
/// * p: projection slice
/// * w: workspace slice
/// * rows: number of rows
/// * cols: number of cols
/// * stride: stride of the data
fn lapply_householder(
    h: &mut [f32],
    p: &mut [f32],
    w: &mut [f32],
    tau: f32,
    rows: usize,
    cols: usize,
    stride: usize,
) {
    // println!("p {p:?}");
    debug_assert!(cols <= w.len());
    debug_assert_eq!(rows, p.len());
    // (I - tuu')A;
    // A -= t*uu'A;
    // w := u'A;
    // R -= t*uw';
    let mut roffset = 0;
    for j in 0..cols {
        // let scalar = p[0];
        // scalar implicitly 1
        w[j] = h[j];
    }
    for i in 1..rows {
        roffset += stride;
        let scalar = p[i];
        for j in 0..cols {
            w[j] += scalar * h[roffset + j];
        }
    }
    for j in 0..cols {
        w[j] *= tau;
        h[j] -= w[j];
    }
    roffset = 0;
    for i in 1..rows {
        roffset += stride;
        for j in 0..cols {
            h[roffset + j] -= p[i] * w[j];
        }
    }
}
/// rapply_householder
///
/// applies the transformation directly starting here to apply
/// to columns 1..cols, simply index into the data and then
/// stride = cols
/// cols = cols - 1;
///
/// * h: hessenberg matrix data
/// * p: projection vector
/// * w: workspace vector
/// * rows: number of rows
/// * cols: number of cols
/// * stride: stride of the data
fn rapply_householder(
    h: &mut [f32],
    p: &mut [f32],
    w: &mut [f32],
    tau: f32,
    rows: usize,
    cols: usize,
    stride: usize,
) {
    debug_assert!(rows <= w.len());
    debug_assert_eq!(cols, p.len());
    // A(I - tuu');
    // A - t*Auu';
    // w := Au;
    // R -= t*wu;
    let mut roffset = 0;
    for i in 0..rows {
        w[i] = h[roffset];
        for k in 1..cols {
            w[i] += h[roffset + k] * p[k];
        }
        w[i] *= tau;
        roffset += stride;
    }
    roffset = 0;
    for i in 0..rows {
        h[roffset] -= w[i];
        for j in 1..cols {
            h[roffset + j] -= w[i] * p[j];
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
        for o in 1..rows {
            active_range -= 1;
            split_range -= 1;
            let (slice, t) = h.split_at_mut(offset + stride);
            let slice = &mut slice[offset + o..offset + cols];
            let proj = &mut p[..split_range];
            let tau = params(slice, proj);
            offset += stride;
            if tau == 0f32 {
                continue;
            }
            lapply_householder(&mut r[offset..], proj, w, tau, active_range, cols, stride);
            rapply_householder(&mut t[o..], proj, w, tau, rows - o, split_range, stride);
            lapply_householder(&mut h[offset..], proj, w, tau, active_range, cols, stride);
        }
    }
}
fn decomp_cpx(h: &mut [f32], w: &mut [f32], mut range: usize, size: usize, mut stride: usize) {
    let s = range * stride;
    let mut e1 = s.saturating_sub(stride + 1);
    let mut e2 = s.saturating_sub(stride + stride + 2);
    let mut i = 0;
    let he1 = h[e1];
    let he2 = h[e2];
    let p = &mut [0f32; 3];
    println!("(r:{range}, e1:{he1}, e2:{he2})");
    while range > 1 && i < 40 {
        println!("iter {i:?}");
        i += 1;
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
            francis_iteration_cpx(h, p, w, size, range, stride);
        }
        let he1 = h[e1];
        let he2 = h[e2];
        println!("(r:{range}, e1:{he1}, e2:{he2})");
    }
}
/// francis_iteration_cpx
///
/// * h: hessenberg linearized matrix
/// * p: projection slice
/// * w: workspace slice
/// * size: static number of rows for rotations
/// * range: number of rows in active window
/// * stride: stride of the data format
fn francis_iteration_cpx(
    h: &mut [f32],
    p: &mut [f32],
    w: &mut [f32],
    size: usize,
    range: usize,
    stride: usize,
) {
    // u1 = a + bi;
    // u2 = a - bi;
    // M = H^2 - H(u1 + u2) +Iu1 *u2;
    // M = H^2 - H *trace +I * det;
    let card = stride * range;
    let tl = card.saturating_sub(stride + 2);
    let bl = card.saturating_sub(2);

    let (m00, m01) = (h[tl], h[tl + 1]);
    let (m10, m11) = (h[bl], h[bl + 1]);

    let (h00, h01) = (h[0], h[1]);
    let (h10, h11) = (h[stride], h[stride + 1]);
    let h12 = h[stride + 2];

    let trace = m00 + m11;
    let deter = m00 * m11 - m01 * m10;

    w[0] = h00 * h00 + h01 * h10 - trace * h00 + deter;
    w[1] = h01 * (h00 + h11 - trace);
    w[2] = h01 * h12;
    // w[1] = h01 * (h00 + h11 - trace);
    // w[2] = h01 * h12;
    let bound = range.min(3);
    let p = &mut p[..bound];
    let tau = params(&mut w[..bound], p);
    if tau != 0f32 {
        rapply_householder(h, p, w, tau, size, bound, stride);
        lapply_householder(h, p, w, tau, bound, range, stride);
    }
    let mut offset = 0;
    // let data = NdArray {
    //     dims:vec![size, size],
    //     data:h.to_vec(),
    // };
    // println!("rotating{data:?}");
    // println!("-------------------------------");
    // println!("-------------------------------");
    // for o in 0..range.saturating_sub(2) {
    // for o in 1..range.saturating_sub(1) {
    for o in 1..range.saturating_sub(1) {
        println!("---------");
        let bound = bound.min(stride - o);
        let (slice, t) = h.split_at_mut(offset + stride);
        let slice = &mut slice[offset + o..offset + o + bound];
        let proj = &mut p[..bound];
        println!("slice {slice:?}");
        let tau = params(slice, proj);
        offset += stride;
        if tau == 0f32 {
            println!("EARLY EXIT  (tau: {tau:?}, cont_o: {o:}, proj: {proj:?}");
            continue;
        }
        // println!("size -o - 1 {}", size - o - 1);
        rapply_householder(&mut t[o..], proj, w, tau, size - o, bound, stride);
        let data = NdArray {
            dims:vec![size, size],
            data:h.to_vec(),
        };
        println!("rapply\n{data:?}");
        println!("---------");
        // lapply_householder(&mut h[offset..], proj, w, tau, bound, range, stride);
        lapply_householder(&mut h[offset..], proj, w, tau, bound, size, stride);
        let data = NdArray {
            dims:vec![size, size],
            data:h.to_vec(),
        };
        println!("lapply\n{data:?}");
        println!("---------");
    }
}
fn eigen(m00: f32, m01: f32, m10: f32, m11: f32) -> f32 {
    let d = (m00 - m11) / 2f32;
    let mut discriminate = d * d + m10 * m01;
    if discriminate >= -EPSILON {
        m11 + d - d.signum() * discriminate.max(0f32).sqrt()
    } else {
        println!("complex discriminate {discriminate:?}");
        m11 + d
    }
}
// fn eigen(m00: f32, m01: f32, m10: f32, m11: f32) -> f32 {
//     let d = (m00 - m11) / 2f32;
//     m11 + d - d.signum() * (d * d + m10 * m10).sqrt()
// }
// h:= hessenberg matrix
fn decomp_sym(h: &mut [f32], mut range: usize, size: usize, mut stride: usize) {
    let s = range * stride;
    let mut e1 = s.saturating_sub(stride + 1);
    let mut e2 = s.saturating_sub(stride + stride + 2);
    let mut i = 0;
    let he1 = h[e1];
    let he2 = h[e2];
    println!("(r:{range}, e1:{he1}, e2:{he2})");
    while range > 1 && i < 40 {
        println!("iter {i:?}");
        i += 1;
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
            francis_iteration_sym(h, size, range, stride);
        }
        let he1 = h[e1];
        let he2 = h[e2];
        println!("(r:{range}, e1:{he1}, e2:{he2})");
    }
}
fn francis_iteration_sym(h: &mut [f32], size: usize, range: usize, stride: usize) {
    let card = stride * range;
    let tl = card.saturating_sub(stride + 2);
    let bl = card.saturating_sub(2);
    let eig = eigen(h[tl], h[tl + 1], h[bl], h[bl + 1]);
    let (_, cosine, sine) = implicit_givens_rotation(h[0] - eig, h[1]);
    apply_g_right(h, 0, 1, stride, range, cosine, -sine);
    apply_gt_left(h, 0, 1, stride, range, cosine, -sine);
    for k in 0..range.saturating_sub(2) {
        let r = k * stride;
        let s1 = k + 1;
        let s2 = k + 2;
        let temp = NdArray {
            dims: vec![range, range],
            data: h.to_vec(),
        };
        let (_, cosine, sine) = implicit_givens_rotation(h[r + s1], h[r + s2]);
        apply_g_right(&mut h[r..], s1, s2, stride, size - k, cosine, -sine);
        // apply_gt_left(h, s1, s2, stride, range, cosine, -sine);
        apply_gt_left(h, s1, s2, stride, range.min(s2+2), cosine, -sine);

        // apply_g_right(&mut h[r..], s1, s2, stride, range - k, cosine, -sine);
        // apply_gt_left(h, s1, s2, stride, range, cosine, -sine);
        // // apply_gt_left(&mut t[r..], k, k + 1, stride, card, cosine, sine);
    }
}
fn generate_symmetric_vector(n: usize) -> Vec<f32> {
    let a = generate_random_matrix(n, n);
    matrix_mult(&a, &a.transpose()).data
}
fn check_hessen_sym() {
    let (rows, cols) = (5, 5);
    let stride = 5;
    let mut h = generate_symmetric_vector(rows);
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
    let reconstruct = matrix_mult(&rotation.transpose(), &reconstruct);
    println!("reconstruct {reconstruct:?}");
}
fn check_hessen() {
    let (rows, cols) = (5, 5);
    let stride = 5;
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
fn check_iteration_sym() -> NdArray {
    let c = 6;
    let (rows, cols) = (c, c);
    let stride = c;
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
    println!("hessenberg {kernel:?}");
    let rotation = NdArray {
        dims: vec![rows, cols],
        data: r.clone(),
    };
    francis_iteration_sym(&mut h, rows, rows, stride);
    let output = NdArray {
        dims: vec![rows, cols],
        data: h.clone(),
    };
    println!("final {output:?}");
    output
}
fn check_decomp_sym() -> NdArray {
    let c = 4;
    let (rows, cols) = (c, c);
    let stride = c;
    let mut h = generate_symmetric_vector(rows);
    // let mut h = generate_random_vector(rows * cols);
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
    println!("hessenberg {kernel:?}");
    decomp_sym(&mut h, c, c, c);
    // francis_iteration(&mut h, rows, stride);
    let output = NdArray {
        dims: vec![rows, cols],
        data: h.clone(),
    };
    println!("final {output:?}");
    output
}
fn check_iteration_cpx() -> NdArray {
    let c = 4;
    let (rows, cols) = (c, c);
    let stride = c;
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
    println!("hessenberg {kernel:?}");
    println!("-------------------------------");
    println!("-------------------------------");
    let rotation = NdArray {
        dims: vec![rows, cols],
        data: r.clone(),
    };
    let mut p = [0f32; 3];
    w.fill(0f32);
    francis_iteration_cpx(&mut h, &mut p, &mut w, rows, rows, stride);
    let output = NdArray {
        dims: vec![rows, cols],
        data: h.clone(),
    };
    println!("final {output:?}");
    output
}
fn check_decomp_cpx() -> NdArray {
    let c = 5;
    let (rows, cols) = (c, c);
    let stride = c;
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
    println!("hessenberg {kernel:?}");
    w.fill(0f32);
    decomp_cpx(&mut h, &mut w, c, c, c);
    // francis_iteration(&mut h, rows, stride);
    let output = NdArray {
        dims: vec![rows, cols],
        data: h.clone(),
    };
    println!("final {output:?}");
    output
}

fn main() {
    // check_decomp_sym();
    check_decomp_cpx();
    // check_iteration_cpx();
    // check_hessen_sym();
    // check_iteration_sym();
    // check_hessen();
    // test_orthogonal();
}
