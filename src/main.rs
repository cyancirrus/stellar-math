#![allow(unused)]
use stellar::algebra::ndmethods::create_identity_vector;
use stellar::algebra::ndmethods::matrix_mult;
use stellar::equality::approximate::approx_vector_eq;
use stellar::random::generation::generate_random_vector;
use stellar::structure::ndarray::NdArray;

use stellar::decomposition::svd::bidiagonalization::{lbidiagonal, ubidiagonal};
use stellar::decomposition::svd::bulge_chasing::{decomp_usym, decomp_lsym};
#[rustfmt::skip]
use stellar::decomposition::svd::verify::{
    full_ubidiagonal,
    full_lbidiagonal,
    full_decomp_ugivens,
    full_decomp_lgivens,
    full_decomp_usym,
full_decomp_lsym,
};

fn diagonal(b: &[f32], card: usize, stride: usize) -> Vec<f32> {
    (0..card).map(|i| b[i * stride + i]).collect()
}

fn thing_upper() {
    let mut max_iters = 40;
    let mut tolerance = 1e-6;
    let mut absolute = 1e-5;

    let rows = 12;
    let cols = 6;
    let card = rows.min(cols);
    let stride = cols;
    let maximum = rows.max(cols);
    let mut b = generate_random_vector(rows * cols);
    let mut p = vec![0f32; maximum];
    let mut w = vec![0f32; maximum];

    ubidiagonal(&mut b, &mut p, &mut w, rows, cols, card, stride);
    let bmat = NdArray {
        dims: vec![rows, cols],
        data: b.clone(),
    };
    println!("bidiagonal {bmat:?}");

    decomp_usym(&mut b, card, stride, max_iters, tolerance, absolute);
    let smat = NdArray {
        dims: vec![rows, cols],
        data: b.clone(),
    };
    println!("singular {smat:?}");
}

fn full_upper() {
    let mut max_iters = 40;
    let mut tolerance = 1e-6;
    let mut absolute = 1e-5;

    let rows = 12;
    let cols = 6;
    let card = rows.min(cols);
    let stride = cols;
    let maximum = rows.max(cols);
    let mut b = generate_random_vector(rows * cols);
    let mut p = vec![0f32; maximum];
    let mut w = vec![0f32; maximum];
    let mut u = create_identity_vector(rows, rows);
    let mut v = create_identity_vector(cols, cols);
    let input = NdArray {
        dims: vec![rows, cols],
        data: b.clone(),
    };

    full_ubidiagonal(
        &mut b, &mut u, &mut v, &mut p, &mut w, rows, cols, card, stride,
    );
    let bmat = NdArray {
        dims: vec![rows, cols],
        data: b.clone(),
    };
    println!("bidiagonal {bmat:?}");

    full_decomp_usym(
        &mut b, &mut u, &mut v, rows, cols, card, stride, max_iters, tolerance, absolute,
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

    println!("singular {singular:?}");

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
    println!("uut {uut:?}");
    println!("vvt {vvt:?}");
    println!("original {input:?}");

    // U Sigma V' ~= original
    let reconstruct = matrix_mult(&umat, &singular);
    let reconstruct = matrix_mult(&reconstruct, &vmat.transpose());
    println!("reconstruct {reconstruct:?}");
}

fn test_lower() {
    let mut max_iters = 40;
    let mut tolerance = 1e-6;
    let mut absolute = 1e-5;

    let rows = 6;
    let cols = 8;
    let card = rows.min(cols);
    let stride = cols;
    let maximum = rows.max(cols);
    let mut b = generate_random_vector(rows * cols);
    let mut p = vec![0f32; maximum];
    let mut w = vec![0f32; maximum];

    lbidiagonal(&mut b, &mut p, &mut w, rows, cols, card, stride);
    let bmat = NdArray {
        dims: vec![rows, cols],
        data: b.clone(),
    };
    println!("bidiagonal {bmat:?}");

    decomp_lsym(&mut b, card, stride, max_iters, tolerance, absolute);
    let smat = NdArray {
        dims: vec![rows, cols],
        data: b.clone(),
    };
    println!("singular {smat:?}");
}

fn main() {
    let mut max_iters = 40;
    let mut tolerance = 1e-6;
    let mut absolute = 1e-5;

    let rows = 6;
    let cols = 12;
    let card = rows.min(cols);
    let stride = cols;
    let maximum = rows.max(cols);
    let mut b = generate_random_vector(rows * cols);
    let mut p = vec![0f32; maximum];
    let mut w = vec![0f32; maximum];
    let mut u = create_identity_vector(rows, rows);
    let mut v = create_identity_vector(cols, cols);
    let input = NdArray {
        dims: vec![rows, cols],
        data: b.clone(),
    };

    full_lbidiagonal(
        &mut b, &mut u, &mut v, &mut p, &mut w, rows, cols, card, stride,
    );
    let bmat = NdArray {
        dims: vec![rows, cols],
        data: b.clone(),
    };
    println!("bidiagonal {bmat:?}");

    full_decomp_lsym(
        &mut b, &mut u, &mut v, rows, cols, card, stride, max_iters, tolerance, absolute,
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

    println!("singular {singular:?}");

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
    println!("uut {uut:?}");
    println!("vvt {vvt:?}");
    println!("original {input:?}");

    // U Sigma V' ~= original
    let reconstruct = matrix_mult(&umat, &singular);
    let reconstruct = matrix_mult(&reconstruct, &vmat.transpose());
    println!("reconstruct {reconstruct:?}");
}
