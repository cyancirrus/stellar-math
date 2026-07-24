use crate::decomposition::francis::constants::{MAX_ITERS, TOLERANCE};
use crate::decomposition::sgivens::{apply_g_left, apply_g_right, apply_gt_right, apply_gt_left, implicit_givens_rotation};
use crate::structure::ndarray::NdArray;
#[rustfmt::skip]
use crate::decomposition::francis::primitives::{
    params,
    deflate,
    eigen,
    double_shift,
    exception_shift,
    complex_eig_pair,
    lapply_householder,
    rapply_householder,
};

#[rustfmt::skip]
pub fn full_decomp_sym(
    h: &mut [f32],
    r: &mut [f32],
    mut range: usize,
    size: usize,
    stride: usize
) {
    let s = range * stride;
    let mut e1 = s.saturating_sub(stride + 1);
    let mut e2 = s.saturating_sub(stride + stride + 2);
    let mut tl = s.saturating_sub(stride + 2);
    let mut bl = s.saturating_sub(2);
    let mut curriter = 0;
    let _he1 = h[e1];
    let _he2 = h[e2];
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
        // } else if h[e2].abs() < TOLERANCE {
        //     deflate(
        //         2,
        //         stride,
        //         &mut range,
        //         &mut e1,
        //         &mut e2,
        //         &mut tl,
        //         &mut bl,
        //         &mut curriter,
        //     );
        } else {
            full_francis_iteration_sym(h, r, size, range, stride, tl, bl);
        }
    }
    println!("range {range:?}");
    if range > 1 {
        println!("fail");
    }
}
pub fn full_decomp_cpx(
    h: &mut [f32],
    r: &mut [f32],
    w: &mut [f32],
    mut range: usize,
    size: usize,
    stride: usize,
) {
    let s = range * stride;
    let mut e1 = s.saturating_sub(stride + 1);
    let mut e2 = s.saturating_sub(stride + stride + 2);
    let mut tl = s.saturating_sub(stride + 2);
    let mut bl = s.saturating_sub(2);
    let mut curriter = 0;
    let _he1 = h[e1];
    let _he2 = h[e2];
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
                full_francis_iteration_cpx_2x2(h, r, size, stride, tl, bl);
            // } else if stall > 0 && (stall + 4) % 10 == 0 {
            } else if (stall + 8) % 12 == 0 {
                // } else if (stall + 4) % 10 == 0 {
                // } else if stall == 6 {
                exception_shift(h, w, stride, range, tl, bl);
                full_francis_iteration_cpx(h, r, p, w, size, range, stride, tl, bl);
            } else {
                double_shift(h, w, stride, range, tl, bl);
                full_francis_iteration_cpx(h, r, p, w, size, range, stride, tl, bl);
            }
            stall += 1;
        }
    }
    if range > 1 {
        println!("missed");
    }
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
pub fn full_francis_iteration_cpx(
    h: &mut [f32],
    r: &mut [f32],
    p: &mut [f32],
    w: &mut [f32],
    size: usize,
    range: usize,
    stride: usize,
    _tl: usize,
    _bl: usize,
) {
    let bound = range.min(3);
    let p = &mut p[..bound];
    let tau = params(&mut w[..bound], p);
    if tau != 0f32 {
        rapply_householder(h, p, w, tau, size, bound, stride);
        lapply_householder(h, p, w, tau, bound, range, stride);
        // ----------------- tracking the rotation matrix
        // rapply_householder(r, p, w, tau, size, bound, stride);
        lapply_householder(r, p, w, tau, bound, size, stride);
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
        lapply_householder(&mut r[offset..], proj, w, tau, bound, size, stride);
        // rapply_householder(&mut r[o..], proj, w, tau, size - o, bound, stride);
    }
}
pub fn full_francis_iteration_cpx_2x2(
    h: &mut [f32],
    r: &mut [f32],
    size: usize,
    stride: usize,
    tl: usize,
    bl: usize,
) {
    let eig = eigen(h[tl], h[tl + 1], h[bl], h[bl + 1]);
    let (_, cosine, sine) = implicit_givens_rotation(h[0] - eig, h[1]);
    apply_g_right(h, 0, 1, stride, size, cosine, -sine);
    apply_gt_left(h, 0, 1, stride, 2, cosine, -sine);
}
/// full_francis_iteration_sym
///
/// * h: hessenberg linearized matrix
/// * r: rotation accumulated linearized matrix
/// * size: static number of rows for rotations
/// * range: number of rows in active window
/// * stride: stride of the data format
/// * tl: top left of the window for the eigens
/// * bl: bottom left of the window for the eigens
pub fn full_francis_iteration_sym(
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
    apply_gt_right(h, 0, 1, stride, size, cosine, sine);
    apply_g_left(h, 0, 1, stride, range, cosine, sine);
    apply_g_left(r, 0, 1, stride, size, cosine, sine);
    for o in 0..range.saturating_sub(2) {
        let row = o * stride;
        let s1 = o + 1;
        let s2 = o + 2;
        let _temp = NdArray {
            dims: vec![range, range],
            data: h.to_vec(),
        };
        let (_, cosine, sine) = implicit_givens_rotation(h[row + s1], h[row + s2]);
        apply_gt_right(&mut h[row..], s1, s2, stride, range - o, cosine, sine);
        apply_g_left(h, s1, s2, stride, range, cosine, sine);
        // apply_g_right(&mut h[row..], s1, s2, stride, range - o, cosine, -sine);
        // apply_gt_left(h, s1, s2, stride, range, cosine, -sine);
        apply_g_left(r, s1, s2, stride, size, cosine, sine);
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
mod test_hessenberg_reconstructions {
    use super::*;
    use crate::algebra::ndmethods::matrix_mult;
    use crate::equality::approximate::{approx_vector_eq, approx_vector_tol_eq};
    use crate::random::generation::{
        generate_identity_vector, generate_random_matrix, generate_random_vector,
        generate_symmetric_vector,
    };

    #[test]
    fn test_hessenberg_reconstruct_general() {
        for dim in [1, 2, 4, 7] {
            let (rows, cols) = (dim, dim);
            let stride = dim;
            let mut h = generate_random_vector(rows * cols);
            let mut r = generate_identity_vector(rows, cols);
            let mut p = vec![0f32; cols];
            let mut w = vec![0f32; rows];
            let original = NdArray {
                dims: vec![rows, cols],
                data: h.clone(),
            };
            full_hessenberg(&mut h, &mut r, &mut p, &mut w, rows, cols, stride);
            let kernel = NdArray {
                dims: vec![rows, cols],
                data: h.clone(),
            };
            let rotation = NdArray {
                dims: vec![rows, cols],
                data: r.clone(),
            };
            // R R' ~= I
            let rrt = matrix_mult(&rotation, &rotation.transpose());
            let identity = generate_identity_vector(rows, cols);
            assert!(
                approx_vector_eq(&rrt.data, &identity),
                "dim={dim}: R R' not orthogonal, got {:?}",
                rrt.data
            );
            // R' R ~= I
            let rtr = matrix_mult(&rotation.transpose(), &rotation);
            assert!(
                approx_vector_eq(&rtr.data, &identity),
                "dim={dim}: R' R not orthogonal, got {:?}",
                rtr.data
            );
            // R' H R ~= original
            let reconstruct = matrix_mult(&rotation.transpose(), &matrix_mult(&kernel, &rotation));
            assert!(
                approx_vector_eq(&reconstruct.data, &original.data),
                "dim={dim}: reconstruction mismatch, got {:?} expected {:?}",
                reconstruct.data,
                original.data
            );
        }
    }
    #[test]
    fn test_hessenberg_reconstruct_symmetric() {
        for dim in [1, 2, 4, 7] {
            let (rows, cols) = (dim, dim);
            let stride = dim;
            let mut h = generate_symmetric_vector(dim);
            let mut r = generate_identity_vector(rows, cols);
            let mut p = vec![0f32; cols];
            let mut w = vec![0f32; rows];
            let original = NdArray {
                dims: vec![rows, cols],
                data: h.clone(),
            };
            full_hessenberg(&mut h, &mut r, &mut p, &mut w, rows, cols, stride);
            let kernel = NdArray {
                dims: vec![rows, cols],
                data: h.clone(),
            };
            let rotation = NdArray {
                dims: vec![rows, cols],
                data: r.clone(),
            };
            let identity = generate_identity_vector(rows, cols);
            let rrt = matrix_mult(&rotation, &rotation.transpose());
            assert!(
                approx_vector_eq(&rrt.data, &identity),
                "dim={dim}: R R' not orthogonal"
            );
            // symmetric-specific: hessenberg of a symmetric matrix should be
            // tridiagonal, i.e. zero below the first subdiagonal
            for i in 0..rows {
                for j in 0..cols {
                    if i > j + 1 {
                        assert!(
                            h[i * stride + j].abs() < 1e-2,
                            "dim={dim}: expected tridiagonal, got h[{i}][{j}]={}",
                            h[i * stride + j]
                        );
                    }
                }
            }
            // R' H R ~= original
            let reconstruct = matrix_mult(&rotation.transpose(), &matrix_mult(&kernel, &rotation));
            assert!(
                approx_vector_eq(&reconstruct.data, &original.data),
                "dim={dim}: symmetric reconstruction mismatch"
            );
        }
    }
}
