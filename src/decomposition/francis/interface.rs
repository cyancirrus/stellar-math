use crate::algebra::ndmethods::basic_mult;
use crate::algebra::ndmethods::create_identity_matrix;
use crate::algebra::ndmethods::matrix_mult;
use crate::decomposition::lower_upper::LuPivotDecompose;
use crate::decomposition::lq::AutumnDecomp;
use crate::decomposition::schur::real_schur;
use crate::decomposition::francis::constants::{EPSILON, TOLERANCE, MAX_ITERS};
use crate::decomposition::sgivens::{
    apply_g_left, apply_g_right, apply_gt_left, apply_gt_right, implicit_givens_rotation,
};
use crate::equality::approximate::approx_vector_tol_eq;
use crate::random::generation::{
    generate_identity_vector, generate_random_matrix, generate_random_vector,
};
use crate::structure::ndarray::NdArray;
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
    debug_assert_eq!(v.len(), w.len());
    let mut max_element = 0f32;
    for val in v.iter() {
        let v = val.abs();
        if v > max_element {
            max_element = v
        };
    }
    if max_element.abs() < EPSILON {
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
    let g = w[0].signum() * magnitude_squared.sqrt();
    let scale = w[0] + g;
    let inv_scale = 1f32 / scale;
    for val in w[1..].iter_mut() {
        *val *= inv_scale;
    }
    v[0] = -g * max_element;
    w[0] = 1f32;
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
fn deflate(
    amount: usize,
    stride: usize,
    range: &mut usize,
    e1: &mut usize,
    e2: &mut usize,
    tl: &mut usize,
    bl: &mut usize,
    // stall: &mut usize,
    curriter: &mut usize,
) {
    let shift = amount * stride + amount;
    *range -= amount;
    *e1 = e1.saturating_sub(shift);
    *e2 = e2.saturating_sub(shift);
    *tl = tl.saturating_sub(shift);
    *bl = bl.saturating_sub(shift);
    // *stall = 0;
    *curriter = curriter.saturating_sub(MAX_ITERS >> 1);
}

fn complex_eig_pair(h: &mut [f32], tl: usize, bl: usize) -> bool {
    let d = (h[tl] - h[bl + 1]) / 2f32;
    d * d + h[tl + 1] * h[bl] < EPSILON
}
fn full_decomp_cpx(
    h: &mut [f32],
    r: &mut [f32],
    w: &mut [f32],
    mut range: usize,
    size: usize,
    mut stride: usize,
) {
    let s = range * stride;
    let mut e1 = s.saturating_sub(stride + 1);
    let mut e2 = s.saturating_sub(stride + stride + 2);
    let mut tl = s.saturating_sub(stride + 2);
    let mut bl = s.saturating_sub(2);
    let mut curriter = 0;
    let he1 = h[e1];
    let he2 = h[e2];
    let p = &mut [0f32; 3];
    let mut stall = 0;
    while range > 0 && curriter < MAX_ITERS {
        curriter += 1;
        if h[e1].abs() < TOLERANCE {
            stall = 0;
            deflate(
                1,
                stride,
                &mut range,
                &mut e1,
                &mut e2,
                &mut tl,
                &mut bl,
                &mut curriter,
            );
        } else if h[e2].abs() < TOLERANCE {
            // if e2 == 0 then we are hitting eigen which should be greater than tolerance
            deflate(
                2,
                stride,
                &mut range,
                &mut e1,
                &mut e2,
                &mut tl,
                &mut bl,
                &mut curriter,
            );
            stall = 0;
        } else if range == 2 && complex_eig_pair(h, tl, bl) {
            deflate(
                2,
                stride,
                &mut range,
                &mut e1,
                &mut e2,
                &mut tl,
                &mut bl,
                &mut curriter,
            );
            stall = 0;
        } else {
            if range == 2 {
                francis_iteration_cpx_2x2(h, size, stride, tl, bl);
            // } else if stall > 0 && (stall + 4) % 10 == 0 {
            } else if (stall + 8) % 12 == 0 {
                // } else if (stall + 4) % 10 == 0 {
                // } else if stall == 6 {
                exception_shift(h, w, stride, range, tl, bl);
                francis_iteration_cpx(h, p, w, size, range, stride, tl, bl);
            } else {
                double_shift(h, w, stride, range, tl, bl);
                francis_iteration_cpx(h, p, w, size, range, stride, tl, bl);
            }
            stall += 1;
        }
    }
    if range > 1 {
        println!("missed");
    }
}
fn decomp_cpx(h: &mut [f32], w: &mut [f32], mut range: usize, size: usize, mut stride: usize) {
    let s = range * stride;
    let mut e1 = s.saturating_sub(stride + 1);
    let mut e2 = s.saturating_sub(stride + stride + 2);
    let mut tl = s.saturating_sub(stride + 2);
    let mut bl = s.saturating_sub(2);
    let mut curriter = 0;
    let he1 = h[e1];
    let he2 = h[e2];
    let p = &mut [0f32; 3];
    let mut stall = 0;
    while range > 0 && curriter < MAX_ITERS {
        curriter += 1;
        if h[e1].abs() < TOLERANCE {
            stall = 0;
            deflate(
                1,
                stride,
                &mut range,
                &mut e1,
                &mut e2,
                &mut tl,
                &mut bl,
                &mut curriter,
            );
        } else if h[e2].abs() < TOLERANCE {
            // if e2 == 0 then we are hitting eigen which should be greater than tolerance
            deflate(
                2,
                stride,
                &mut range,
                &mut e1,
                &mut e2,
                &mut tl,
                &mut bl,
                &mut curriter,
            );
            stall = 0;
        } else if range == 2 && complex_eig_pair(h, tl, bl) {
            deflate(
                2,
                stride,
                &mut range,
                &mut e1,
                &mut e2,
                &mut tl,
                &mut bl,
                &mut curriter,
            );
            stall = 0;
        } else {
            if range == 2 {
                francis_iteration_cpx_2x2(h, size, stride, tl, bl);
            // } else if stall > 0 && (stall + 4) % 10 == 0 {
            } else if (stall + 8) % 12 == 0 {
                // } else if (stall + 4) % 10 == 0 {
                // } else if stall == 6 {
                exception_shift(h, w, stride, range, tl, bl);
                francis_iteration_cpx(h, p, w, size, range, stride, tl, bl);
            } else {
                double_shift(h, w, stride, range, tl, bl);
                francis_iteration_cpx(h, p, w, size, range, stride, tl, bl);
            }
            stall += 1;
        }
    }
    if range > 1 {
        println!("missed");
    }
}
/// double_shift
///   - standard shift for francis iteration
///
/// * h: hessenberg linearized matrix
/// * p: projection slice
/// * w: workspace slice
/// * range: number of rows in active window
/// * stride: stride of the data format
fn double_shift(h: &mut [f32], w: &mut [f32], stride: usize, range: usize, tl: usize, bl: usize) {
    // u1 = a + bi;
    // u2 = a - bi;
    // M = H^2 - H(u1 + u2) +Iu1 *u2;
    // M = H^2 - H *trace +I * det;
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
}
/// exception_shift
///   - standard shift for francis iteration
///
/// * h: hessenberg linearized matrix
/// * p: projection slice
/// * w: workspace slice
/// * range: number of rows in active window
/// * stride: stride of the data format
fn exception_shift(
    h: &mut [f32],
    w: &mut [f32],
    stride: usize,
    range: usize,
    tl: usize,
    bl: usize,
) {
    // u1 = a + bi;
    // u2 = a - bi;
    // M = H^2 - H(u1 + u2) +Iu1 *u2;
    // M = H^2 - H *trace +I * det;
    let (m00, m01) = (h[tl], h[tl + 1]);
    let (m10, m11) = (h[bl], h[bl + 1]);

    let (h00, h01) = (h[0], h[1]);
    let (h10, h11) = (h[stride], h[stride + 1]);
    let h12 = h[stride + 2];

    let s = m01.abs() + h01.abs();
    let trace = 2.0 * s;
    let deter = s * s;

    w[0] = h00 * h00 + h01 * h10 - trace * h00 + deter;
    w[1] = h01 * (h00 + h11 - trace);
    w[2] = h01 * h12;
}

/// francis_iteration_cpx
///
/// * h: hessenberg linearized matrix
/// * r: rotaiton linearized matrix
/// * p: projection slice
/// * w: workspace slice
/// * size: static number of rows for rotations
/// * range: number of rows in active window
/// * stride: stride of the data format
fn full_francis_iteration_cpx(
    h: &mut [f32],
    r: &mut [f32],
    p: &mut [f32],
    w: &mut [f32],
    size: usize,
    range: usize,
    stride: usize,
    tl: usize,
    bl: usize,
) {
    let bound = range.min(3);
    let p = &mut p[..bound];
    let tau = params(&mut w[..bound], p);
    if tau != 0f32 {
        rapply_householder(h, p, w, tau, size, bound, stride);
        lapply_householder(h, p, w, tau, bound, range, stride);
        // ----------------- tracking the rotation matrix
        rapply_householder(r, p, w, tau, size, bound, stride);
    }
    let mut offset = 0;
    for o in 1..range.saturating_sub(1) {
        let bound = bound.min(stride - o);
        let (slice, t) = h.split_at_mut(offset + stride);
        let slice = &mut slice[offset + o..offset + o + bound];
        let proj = &mut p[..bound];
        let tau = params(slice, proj);
        offset += stride;
        if tau == 0f32 {
            continue;
        }
        rapply_householder(&mut t[o..], proj, w, tau, size - o, bound, stride);
        lapply_householder(&mut h[offset..], proj, w, tau, bound, range, stride);
        // ----------------- tracking the rotation matrix
        rapply_householder(&mut r[o..], proj, w, tau, size - o, bound, stride);
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
    tl: usize,
    bl: usize,
) {
    let bound = range.min(3);
    let p = &mut p[..bound];
    let tau = params(&mut w[..bound], p);
    if tau != 0f32 {
        rapply_householder(h, p, w, tau, size, bound, stride);
        lapply_householder(h, p, w, tau, bound, range, stride);
        // lapply_householder(h, p, w, tau, bound, size, stride);
    }
    let mut offset = 0;
    for o in 1..range.saturating_sub(1) {
        let bound = bound.min(stride - o);
        let (slice, t) = h.split_at_mut(offset + stride);
        let slice = &mut slice[offset + o..offset + o + bound];
        let proj = &mut p[..bound];
        let tau = params(slice, proj);
        offset += stride;
        if tau == 0f32 {
            continue;
        }
        rapply_householder(&mut t[o..], proj, w, tau, size - o, bound, stride);
        // lapply_householder(&mut h[offset..], proj, w, tau, bound, size, stride);
        lapply_householder(&mut h[offset..], proj, w, tau, bound, range, stride);
    }
}
fn francis_iteration_cpx_2x2(h: &mut [f32], size: usize, stride: usize, tl: usize, bl: usize) {
    let eig = eigen(h[tl], h[tl + 1], h[bl], h[bl + 1]);
    let (_, cosine, sine) = implicit_givens_rotation(h[0] - eig, h[1]);
    apply_g_right(h, 0, 1, stride, size, cosine, -sine);
    apply_gt_left(h, 0, 1, stride, 2, cosine, -sine);
}
fn eigen(m00: f32, m01: f32, m10: f32, m11: f32) -> f32 {
    let d = (m00 - m11) / 2f32;
    let mut discriminate = d * d + m10 * m01;
    if discriminate >= -EPSILON {
        m11 + d - d.signum() * discriminate.max(0f32).sqrt()
    } else {
        m11 + d
    }
}
fn decomp_sym(h: &mut [f32], mut range: usize, size: usize, mut stride: usize) {
    let s = range * stride;
    let mut e1 = s.saturating_sub(stride + 1);
    let mut e2 = s.saturating_sub(stride + stride + 2);
    let mut tl = s.saturating_sub(stride + 2);
    let mut bl = s.saturating_sub(2);
    let mut curriter = 0;
    let he1 = h[e1];
    let he2 = h[e2];
    while range > 1 && curriter < MAX_ITERS {
        curriter += 1;
        if h[e1].abs() < TOLERANCE {
            deflate(
                1,
                stride,
                &mut range,
                &mut e1,
                &mut e2,
                &mut tl,
                &mut bl,
                &mut curriter,
            );
        } else if h[e2].abs() < TOLERANCE {
            deflate(
                2,
                stride,
                &mut range,
                &mut e1,
                &mut e2,
                &mut tl,
                &mut bl,
                &mut curriter,
            );
        } else {
            francis_iteration_sym(h, size, range, stride, tl, bl);
        }
    }
}
/// francis_iteration_cpx
///
/// * h: hessenberg linearized matrix
/// * size: static number of rows for rotations
/// * range: number of rows in active window
/// * stride: stride of the data format
/// * tl: top left of the window for the eigens
/// * bl: bottom left of the window for the eigens
fn francis_iteration_sym(
    h: &mut [f32],
    size: usize,
    range: usize,
    stride: usize,
    tl: usize,
    bl: usize,
) {
    let eig = eigen(h[tl], h[tl + 1], h[bl], h[bl + 1]);
    let (_, cosine, sine) = implicit_givens_rotation(h[0] - eig, h[1]);
    apply_g_right(h, 0, 1, stride, range, cosine, -sine);
    apply_gt_left(h, 0, 1, stride, range, cosine, -sine);
    for o in 0..range.saturating_sub(2) {
        let r = o * stride;
        let s1 = o + 1;
        let s2 = o + 2;
        let temp = NdArray {
            dims: vec![range, range],
            data: h.to_vec(),
        };
        let (_, cosine, sine) = implicit_givens_rotation(h[r + s1], h[r + s2]);
        // apply_g_right(&mut h[r..], s1, s2, stride, size - o, cosine, -sine);
        // apply_gt_left(h, s1, s2, stride, range.min(s2 + 2), cosine, -sine);
        apply_g_right(&mut h[r..], s1, s2, stride, range - o, cosine, -sine);
        apply_gt_left(h, s1, s2, stride, range, cosine, -sine);
    }
}
/// full_francis_iteration_cpx
///
/// * h: hessenberg linearized matrix
/// * r: rotation accumulated linearized matrix
/// * size: static number of rows for rotations
/// * range: number of rows in active window
/// * stride: stride of the data format
/// * tl: top left of the window for the eigens
/// * bl: bottom left of the window for the eigens
fn full_francis_iteration_sym(
    h: &mut [f32],
    r: &mut [f32],
    size: usize,
    range: usize,
    stride: usize,
    tl: usize,
    bl: usize,
) {
    let eig = eigen(h[tl], h[tl + 1], h[bl], h[bl + 1]);
    let (_, cosine, sine) = implicit_givens_rotation(h[0] - eig, h[1]);
    apply_g_right(h, 0, 1, stride, range, cosine, -sine);
    apply_gt_left(h, 0, 1, stride, range, cosine, -sine);
    for o in 0..range.saturating_sub(2) {
        let row = o * stride;
        let s1 = o + 1;
        let s2 = o + 2;
        let temp = NdArray {
            dims: vec![range, range],
            data: h.to_vec(),
        };
        let (_, cosine, sine) = implicit_givens_rotation(h[row + s1], h[row + s2]);
        apply_g_right(&mut h[row..], s1, s2, stride, range - o, cosine, -sine);
        apply_gt_left(h, s1, s2, stride, range, cosine, -sine);
        // apply_g_right(&mut h[r..], s1, s2, stride, size - o, cosine, -sine);
        // apply_gt_left(h, s1, s2, stride, range.min(s2 + 2), cosine, -sine);

        // -----------------------
        apply_g_right(&mut r[row..], s1, s2, stride, range - o, cosine, -sine)
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
    full_hessenberg(&mut h, &mut r, &mut p, &mut w, rows, cols, stride);
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
    full_hessenberg(&mut h, &mut r, &mut p, &mut w, rows, cols, stride);
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
    let s = c * c;
    let mut bl = s.saturating_sub(2);
    let mut tl = s.saturating_sub(stride + 2);
    let mut p = vec![0f32; cols];
    let mut w = vec![0f32; rows];
    let input = NdArray {
        dims: vec![rows, cols],
        data: h.clone(),
    };
    println!("before {input:?}");
    full_hessenberg(&mut h, &mut r, &mut p, &mut w, rows, cols, stride);
    let kernel = NdArray {
        dims: vec![rows, cols],
        data: h.clone(),
    };
    println!("hessenberg {kernel:?}");
    let rotation = NdArray {
        dims: vec![rows, cols],
        data: r.clone(),
    };
    francis_iteration_sym(&mut h, rows, rows, stride, tl, bl);
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
    full_hessenberg(&mut h, &mut r, &mut p, &mut w, rows, cols, stride);
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
    let s: usize = c * c;
    let mut bl = s.saturating_sub(2);
    let mut tl = s.saturating_sub(stride + 2);
    let mut h = generate_random_vector(rows * cols);
    let mut r = generate_identity_vector(rows, cols);
    let mut p = vec![0f32; cols];
    let mut w = vec![0f32; rows];
    let input = NdArray {
        dims: vec![rows, cols],
        data: h.clone(),
    };
    println!("before {input:?}");
    full_hessenberg(&mut h, &mut r, &mut p, &mut w, rows, cols, stride);
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
    francis_iteration_cpx(&mut h, &mut p, &mut w, rows, rows, stride, tl, bl);
    let output = NdArray {
        dims: vec![rows, cols],
        data: h.clone(),
    };
    println!("final {output:?}");
    output
}
fn check_decomp_cpx() -> NdArray {
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
    // println!("before {input:?}");
    full_hessenberg(&mut h, &mut r, &mut p, &mut w, rows, cols, stride);
    let kernel = NdArray {
        dims: vec![rows, cols],
        data: h.clone(),
    };
    // println!("hessenberg {kernel:?}");
    w.fill(0f32);
    decomp_cpx(&mut h, &mut w, c, c, c);
    let output = NdArray {
        dims: vec![rows, cols],
        data: h.clone(),
    };
    // println!("final {output:?}");
    output
}

// fn check_reconstruct(n: usize) {

//  RQ = Q'QRQ;
//  --> R * K R'
//  for LQ orientation
//  A == R'KR
//     let mut workspace = vec![0f32; n];
//     let x = generate_random_matrix(n, n);
//     let x = basic_mult(&x, &x.transpose());
//     let det = determinant(x.clone(), &mut workspace);
//     if det.abs() < TOLERANCE {
//         println!("determinant to low\ndet{det:?}");
//         return;
//     }
//     let kernel = x.clone();
//     let nkernel = create_identity_matrix(n);
//     let schur = real_schur(kernel, nkernel, &mut workspace);
//     let q = schur.rotation;
//     let q_star = q.transpose();
//     println!("q {q:?}");
//     let expect = basic_mult(&q, &x);
//     let expect = basic_mult(&expect, &q_star);
//     let result = schur.kernel;
//     println!("expect {expect:?}");
//     println!("result {result:?}");
//     println!("determinant {det:?}");
//     assert!(approx_vector_tol_eq(&result.data, &expect.data, TOLERANCE));
// }
// fn test_orthogonal() {
//     for n in 1..12 {
//         check_orthogonal(n);
//     }
// }
// fn check_orthogonal(n: usize) {
//     let x = generate_random_matrix(n, n);
//     let kernel = x.clone();
//     let nkernel = create_identity_matrix(n);
//     let mut workspace = vec![0f32; n];
//     let schur = real_schur(kernel, nkernel, &mut workspace);
//     let q = schur.rotation;
//     let q_star = q.transpose();
//     println!("q {q:?}");
//     let expect = create_identity_matrix(n);
//     let result = basic_mult(&q, &q_star);
//     println!("expect {expect:?}");
//     println!("result {result:?}");
//     assert!(approx_vector_tol_eq(&result.data, &expect.data, TOLERANCE));
// }

fn main() {
    check_decomp_sym();
    for i in 0..1000 {
        check_decomp_cpx();
        // println!("-----------------");
    }
    // check_iteration_cpx();
    // check_hessen_sym();
    // check_iteration_sym();
    // check_hessen();
    // test_orthogonal();
    // TODO
    // if range > 1 {

    //     let output = NdArray {
    //         dims: vec![size, size],
    //         data: h.to_vec(),
    //     };
    //     println!("-------------- error output inspect----------");
    //     println!("{output:?}");
    //     let (m00, m01) = (h[tl], h[tl + 1]);
    //     let (m10, m11) = (h[bl], h[bl + 1]);
    //     let d = (m00 - m11) / 2.0;
    //     let disc = d * d + m01 * m10;
    //     println!(
    //         "max_iter reached (r:{range}, e1: {}, e2: {}, disc: {disc:?})",
    //         h[e1], h[e2]
    //     );
    // }
}

