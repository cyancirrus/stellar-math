use crate::decomposition::francis::primitives::{
    lapply_householder, params, rapply_householder,
};
use crate::decomposition::sgivens::{
    apply_g_left, apply_gt_right, implicit_givens_rotation,
};
fn full_zero_col(
    b: &mut [f32],
    u: &mut [f32],
    p: &mut [f32],
    w: &mut [f32],
    rows: usize,
    ract: usize,
    cact: usize,
    stride: usize,
) {
    let mut roffset = 0;
    for k in 0..ract {
        w[k] = b[roffset];
        b[roffset] = 0f32;
        roffset += stride;
    }
    let proj = &mut p[..ract];
    let tau = params(&mut w[..ract], proj);
    b[0] = w[0];
    if tau == 0f32 { return; }
    if cact != 0 { 
        lapply_householder(
            &mut b[1..],
            proj,
            w,
            tau,
            ract,
            cact.saturating_sub(1),
            stride,
        );
    }
    rapply_householder(
        u,
        proj,
        w,
        tau,
        rows,
        ract,
        rows,
    );
}
fn full_zero_row(
    b: &mut [f32],
    v: &mut [f32],
    p: &mut [f32],
    w: &mut [f32],
    cols: usize,
    ract: usize,
    cact: usize,
    stride: usize,
) {
    let slice = &mut b[..cact];
    let proj = &mut p[..cact];
    let tau = params(slice, proj);
    if tau == 0f32 { return; }
    if ract != 0 {
        rapply_householder(
            &mut b[stride..],
            proj,
            w,
            tau,
            ract,
            cact,
            stride,
        );
    }
    rapply_householder(
        v,
        proj,
        w,
        tau,
        cols,
        cact,
        cols,
    );
}
/// # full ubidiagonal :: upper bidiagonal
///
/// * b: matrix to create the bidiagonal
/// * u: eigenvectors of AA' ie rowspace
/// * v: eigenvectors of A'A ie colspace
/// * p: projection vector
/// * w: workspace vector
/// * rows: number of rows
/// * cols: number of cols
/// * stride: stride of the data
#[rustfmt::skip]
pub fn full_ubidiagonal(
    b: &mut [f32],
    u: &mut [f32],
    v: &mut [f32],
    p: &mut [f32],
    w: &mut [f32],
    rows: usize,
    cols: usize,
    card: usize,
    stride: usize,
) {
    let mut ract = rows;
    let mut cact = cols;
    let mut offset = 0;
    let pivot = card.saturating_sub(1);
    for k in 0..pivot {
        full_zero_col(&mut b[offset + k..], &mut u[k..], p, w, rows, ract, cact, stride);
        full_zero_row(&mut b[offset + k + 1..], &mut v[k + 1 ..], p, w, cols, ract - 1, cact - 1, stride);
        ract -= 1;
        cact -= 1;
        offset += stride;
    }
    if cact < ract {
        full_zero_col(&mut b[offset + pivot..], &mut u[pivot..], p, w, rows, ract, cact - 1, stride);
    } else if cact > ract {
        full_zero_row(&mut b[offset + pivot + 1..], &mut v[pivot + 1..], p, w, cols, 0, cact-1, stride);
    }
}
/// # full_lbidiagonal :: lower bidiagonal
///
/// * b: matrix to create the bidiagonal
/// * p: projection vector
/// * w: workspace vector
/// * rows: number of rows
/// * cols: number of cols
/// * stride: stride of the data
pub fn full_lbidiagonal(
    b: &mut [f32],
    u: &mut [f32],
    v: &mut [f32],
    p: &mut [f32],
    w: &mut [f32],
    rows: usize,
    cols: usize,
    card: usize,
    stride: usize,
) {
    // rows and active columns
    let mut ract = rows;
    let mut cact = cols;
    let pivot = card.saturating_sub(1);
    let mut offset = 0;
    for k in 0..pivot {
        full_zero_row(&mut b[offset + k..], &mut v[k..], p, w, cols, ract - 1, cact, stride);
        full_zero_col(&mut b[offset + k + stride..], &mut u[k + 1..],  p, w, rows, ract - 1, cact, stride);
        ract -= 1;
        cact -= 1;
        offset += stride;
    }
    if cact < ract {
        full_zero_col(&mut b[offset + pivot + stride ..], &mut u[pivot + 1..], p, w, rows, ract-1, cact, stride);
    } else if cact > ract {
        full_zero_row(&mut b[offset + pivot..], &mut v[pivot..], p, w, cols, 0, cact, stride);
    }
}

#[rustfmt::skip]
pub fn full_decomp_lgivens(
    h: &mut [f32],
    u: &mut [f32],
    v: &mut [f32],
    rows: usize,
    cols: usize, card: usize, stride: usize,
    max_iters:usize,
    threshold: f32,
) {
    let interior = card.saturating_sub(2);
    let mut subdiag_norm = f32::INFINITY;
    for _ in 0..max_iters {
    // for _ in 0..1 {
        if subdiag_norm < threshold { break; }
        subdiag_norm = 0f32;
        let mut offset = 0;
        let mut uoffset = 0;
        let mut voffset = 0;
        // push zero into row
        let (_, cos, sin) = implicit_givens_rotation(h[0], h[stride]);
        apply_g_left(h, 0, 1, stride, 2, cos, sin);
        apply_gt_right(u, 0, 1, rows, rows, cos, sin);
        for _ in 0..interior {
            // push zero into col
            let (_, cos, sin) = implicit_givens_rotation(h[offset], h[offset + 1]);
            apply_gt_right(&mut h[offset ..], 0, 1, stride, 3, cos, sin);
            apply_gt_right(&mut v[voffset ..], 0, 1, cols, cols, cos, sin);
            // push zero into row
            offset += stride;
            voffset += 1;
            uoffset += 1;

            let (_, cos, sin) = implicit_givens_rotation(h[offset], h[offset + stride]);
            apply_g_left(&mut h[offset..], 0, 1, stride, 3, cos, sin);
            apply_gt_right(&mut u[uoffset..], 0, 1, rows, rows, cos, sin);
            subdiag_norm += h[offset].abs();
            offset += 1;
        }
        // // push zero into col
        let (_, cos, sin) = implicit_givens_rotation(h[offset], h[offset + 1]);
        apply_gt_right(&mut h[offset ..], 0, 1, stride, 2, cos, sin);
        apply_gt_right(&mut v[voffset ..], 0, 1, cols, cols, cos, sin);
        subdiag_norm += h[offset + stride].abs();
    }
}
#[rustfmt::skip]
pub fn full_decomp_ugivens(
    h: &mut [f32],
    u: &mut [f32],
    v: &mut [f32],
    rows: usize,
    cols: usize,
    card: usize,
    stride: usize,
    max_iters:usize,
    threshold: f32,
) {
    let interior = card.saturating_sub(2);
    let mut supdiag_norm = f32::INFINITY;
    for _ in 0..max_iters {
        if supdiag_norm < threshold { break; }
        let mut offset = 0;
        let mut uoffset = 0;
        let mut voffset = 0;
        // push zero into col
        let (_, cos, sin) = implicit_givens_rotation(h[0], h[1]);
        apply_gt_right(h, 0, 1, stride, 2, cos, sin);
        apply_gt_right(v, 0, 1, cols, cols, cos, sin);
        for _ in 0..interior {
            // push zero into row
            let (_, cos, sin) = implicit_givens_rotation(h[offset], h[offset + stride]);
            apply_g_left(&mut h[offset..], 0, 1, stride, 3, cos, sin);
            apply_gt_right(&mut u[uoffset..], 0, 1, rows, rows, cos, sin);
            // push zero into col
            offset += 1;
            voffset += 1;
            let (_, cos, sin) = implicit_givens_rotation(h[offset], h[offset + 1]);
            apply_gt_right(&mut h[offset ..], 0, 1, stride, 3, cos, sin);
            apply_gt_right(&mut v[voffset ..], 0, 1, cols, cols, cos, sin);
            supdiag_norm += h[offset].abs();
            uoffset += 1;
            offset += stride;
        }
        // push zero into row
        let (_, cos, sin) = implicit_givens_rotation(h[offset], h[offset + stride]);
        apply_g_left(&mut h[offset..], 0, 1, stride, 2, cos, sin);
        apply_gt_right(&mut u[uoffset..], 0, 1, rows, rows, cos, sin);
        supdiag_norm += h[offset + 1].abs();
    }
}
#[cfg(test)]
mod test_svd_reconstructions {
    use crate::decomposition::svd::interface::{full_svd_decomposition};

    use crate::algebra::ndmethods::matrix_mult;
    use crate::equality::approximate::approx_vector_eq;
    use crate::algebra::ndmethods::create_identity_vector;
    use crate::random::generation::{generate_random_vector};
    use crate::structure::ndarray::NdArray;

    fn check_svd_reconstruct(rows: usize, cols: usize) -> (bool, bool, bool) {
        // returns (u_ortho_ok, v_ortho_ok, reconstruction_ok)
        let card = rows.min(cols);
        let stride = cols;
        let maximum = rows.max(cols);

        let mut u = create_identity_vector(rows, rows);
        let mut v = create_identity_vector(cols, cols);
        let mut w = vec![0f32; maximum];
        let mut p = vec![0f32; maximum];

        let mut b = generate_random_vector(rows * cols);
        let original = NdArray {
            dims: vec![rows, cols],
            data: b.clone(),
        };

        full_svd_decomposition(
            &mut b, &mut u, &mut v, &mut p, &mut w, rows, cols, card, stride, 40, 1e-10,
        );

        let singular = NdArray {
            dims: vec![rows, cols],
            data: b.clone(),
        };
        let umat = NdArray {
            dims: vec![rows, rows],
            data: u.clone(),
        };
        let vmat = NdArray {
            dims: vec![cols, cols],
            data: v.clone(),
        };

        let u_identity = create_identity_vector(rows, rows);
        let v_identity = create_identity_vector(cols, cols);

        // U U' ~= I and U' U ~= I
        let uut = matrix_mult(&umat, &umat.transpose());
        let utu = matrix_mult(&umat.transpose(), &umat);
        let u_ortho_ok =
            approx_vector_eq(&uut.data, &u_identity) && approx_vector_eq(&utu.data, &u_identity);

        // V V' ~= I and V' V ~= I
        let vvt = matrix_mult(&vmat, &vmat.transpose());
        let vtv = matrix_mult(&vmat.transpose(), &vmat);
        let v_ortho_ok =
            approx_vector_eq(&vvt.data, &v_identity) && approx_vector_eq(&vtv.data, &v_identity);

        // U Sigma V' ~= original
        let reconstruct = matrix_mult(&umat, &singular);
        let reconstruct = matrix_mult(&reconstruct, &vmat.transpose());
        let recon_ok = approx_vector_eq(&reconstruct.data, &original.data);

        (u_ortho_ok, v_ortho_ok, recon_ok)
    }

    #[test]
    fn test_svd_reconstruct_square() {
        for dim in [2, 4, 7] {
            let (u_ok, v_ok, r_ok) = check_svd_reconstruct(dim, dim);
            assert!(u_ok, "dim={dim}: U not orthogonal");
            assert!(v_ok, "dim={dim}: V not orthogonal");
            assert!(r_ok, "dim={dim}: reconstruction mismatch");
        }
    }

    #[test]
    fn test_svd_reconstruct_wide() {
        // rows < cols
        for (rows, cols) in [(1, 2), (2, 4), (4, 6), (4, 8)] {
            let (u_ok, v_ok, r_ok) = check_svd_reconstruct(rows, cols);
            assert!(u_ok, "{rows}x{cols}: U not orthogonal");
            assert!(v_ok, "{rows}x{cols}: V not orthogonal");
            assert!(r_ok, "{rows}x{cols}: reconstruction mismatch");
        }
    }

    #[test]
    fn test_svd_reconstruct_tall() {
        // rows > cols
        for (rows, cols) in [(2, 1), (4, 2), (6, 4), (8, 4)] {
            let (u_ok, v_ok, r_ok) = check_svd_reconstruct(rows, cols);
            assert!(u_ok, "{rows}x{cols}: U not orthogonal");
            assert!(v_ok, "{rows}x{cols}: V not orthogonal");
            assert!(r_ok, "{rows}x{cols}: reconstruction mismatch");
        }
    }

    #[rustfmt::skip]
    #[test]
    fn test_svd_reconstruct_trials() {
        let trials = 10_000;
        let mut u_failures = 0;
        let mut v_failures = 0;
        let mut recon_failures = 0;

        for _ in 0..trials {
            let (u_ok, v_ok, r_ok) = check_svd_reconstruct(6, 6);
            if !u_ok { u_failures += 1; }
            if !v_ok { v_failures += 1; }
            if !r_ok { recon_failures += 1; }
        }

        println!("svd: {u_failures} U failures, {v_failures} V failures, {recon_failures} reconstruction failures / {trials}");
        assert!(u_failures < 10, "too many U orthogonality failures: {u_failures}");
        assert!(v_failures < 10, "too many V orthogonality failures: {v_failures}");
        assert!(recon_failures < 10, "too many reconstruction failures: {recon_failures}");
    }
}
